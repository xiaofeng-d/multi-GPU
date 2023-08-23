#!/usr/bin/env python

import argparse

def get_parser():
	parser = argparse.ArgumentParser(description="U-Net for ZA -> Nbody net")
	parser.add_argument("cuda_visible_devices", type=str)
	parser.add_argument("--cuda_cache_path", type=str,
		default="/home/scratch/siyuh/nv_ComputeCache")  # to fix issue with auton lab gpu
	parser.add_argument("-c", "--config_file_path", type=str, default='')
	return parser



if __name__ == "__main__":

	parser = get_parser()
	args = parser.parse_args()
	import functools
	import os
	# import os
	os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
	os.environ["CUDA_CACHE_PATH"] = args.cuda_cache_path

	import json
	import numpy as np
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import lightning.pytorch as pl

	import torch.distributed as dist
	from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

	from torch.distributed.fsdp.fully_sharded_data_parallel import (FullyShardedDataParallel as FSDPP,
																	CPUOffload,
																	MixedPrecision,
																	ShardingStrategy,
																	BackwardPrefetch)

	from torch.distributed.fsdp.wrap import (
		size_based_auto_wrap_policy,
		enable_wrap,
		wrap,
	)

	# from torch.distributed.fsdp.fully_sharded_data_parallel import (
	#     CPUOffload,
	#     BackwardPrefetch,
	# )
	from lightning.pytorch.strategies import FSDPStrategy




	from torch.utils.data import DataLoader
	import time
	from data_utils import SimuData, test_prediction, analysis
	import sys
	#sys.path.insert(0, '/mnt/home/siyuh/Project/Recon/Unet/')
	sys.path.insert(0, '/lcrc/project/cosmo_ai/dongx/Multi-GPU/0.3-0.4-multiGPU/')
	sys.path.insert(0, '/lcrc/project/cosmo_ai/dongx/PM-128-redshift/0.3-0.4/ML-Recon/Unet')
	from uNet import BasicBlock, Lpt2NbodyNet




	with open(args.config_file_path) as f:
		configs = json.load(f)


	base_data_path = configs["base_data_path"]
	output_path = configs["output_path"]

	if configs["is_train"]:
		TrainSet = SimuData(base_data_path,
					configs['train']['data_partition']['label'],
					#configs['train']['data_partition']['hIndex'],
					configs['train']['data_partition']['aug'])
		ValSet = SimuData(base_data_path,
					configs['val']['data_partition']['label'],
					#configs['val']['data_partition']['lIndex'],
					#configs['val']['data_partition']['hIndex'],
					configs['val']['data_partition']['aug'])
		TrainLoader = DataLoader(TrainSet,
					batch_size=configs['train']['batch_size'],
					shuffle=True,
					num_workers=configs['val']['num_workers'])
		ValLoader   = DataLoader(ValSet,
					batch_size=configs['val']['batch_size'],
					shuffle=False,
					num_workers=configs['val']['num_workers'])
	elif configs["is_test"]:
		TestSet = SimuData(base_data_path,
					configs['test']['data_partition']['label'],
					#configs['test']['data_partition']['lIndex'],
					#configs['test']['data_partition']['hIndex'],
					configs['test']['data_partition']['aug'])

		TestLoader  = DataLoader(TestSet,
					batch_size=configs['test']['batch_size'],
					shuffle=False,
					num_workers=configs['test']['num_workers'])

	eval_frequency = configs["train"]["eval_frequency"]
	loss_val = []
	loss_train = []
	iterTime = 0
	best_validation_accuracy = 100

	

    ### set up a trainloader

	class LitUNet(pl.LightningModule):
		def __init__(self):
			super().__init__()
			self.configs = configs
			self.net = Lpt2NbodyNet(BasicBlock)
			# self.net = nn.DataParallel(net)
			# self.net = net.to(device)
			# net = Lpt2NbodyNet(BasicBlock)
			# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
			self.loss =  F.mse_loss
			self.save_dir = configs["output_path"]

			# self.my_auto_wrap_policy = functools.partial(
    		# size_based_auto_wrap_policy, min_num_params=20_000_000
			# )

		def forward(self, x):
			ypred = self.net(x)
			return ypred

		def configure_optimizers(self):
			optimizer = torch.optim.Adam(self.net.parameters(),lr=configs["net_params"]['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=configs["net_params"]['reg'])
			return optimizer
		
		def training_step(self, train_batch, batch_idx):
			
			x, y = train_batch

			# data[0]
			# NetInput = torch.autograd.Variable(data[0],requires_grad=False).to(device)
			Y_pred = self.net(x)
			
			# self.loss = F.MSELoss(Y_pred, data[1])
			loss = self.loss(Y_pred, y)

			## tensorboard logger or something   on_epoch = True, then aggregate one output per epoch
			self.log('train_loss', loss, on_epoch = False)

			return loss 
		
		def test_step(self, test_batch, batch_idx):
			x, y = test_batch
			# x, y = test_batch
			Y_pred = self.net(x)
			loss = self.loss(Y_pred, y)
			Y_pred_np = Y_pred.cpu().detach().numpy()
			save_path = os.path.join(self.save_dir, f"y_pred_{batch_idx}.npy")
			np.save(save_path, Y_pred_np)
			self.log('test_loss', loss, on_epoch = False)
			
			return loss
		
		
		def validation_step(self, val_batch, batch_idx):
			x, y = val_batch
			# x, y = val_batch


	## set up traning and validation data  
	print('how many GPUs?')
	print(torch.cuda.device_count())
	print(os.environ['CUDA_VISIBLE_DEVICES'])
	

	def setup(rank, world_size):
		# os.environ['MASTER_ADDR'] = 'localhost'
		# os.environ['MASTER_PORT'] = '12355'
		if args.environment == 'pbs':
			os.environ['RANK'] = os.environ['PMI_RANK']# global 
			os.environ['LOCAL_RANK'] = os.environ['PMI_LOCAL_RANK'] # local
			os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
			args.world_size = int(os.environ['PMI_SIZE'])
			args.global_rank = int(os.environ['PMI_RANK'])
			args.rank = args.global_rank
			args.local_rank = int(os.environ['PMI_LOCAL_RANK']) # wraparound since LOCAL_RANK is actually global?? WRT/ 
			args.local_size = int(os.environ['PMI_LOCAL_SIZE'])
			args.backend = 'nccl'
			args.num_nodes = args.world_size // args.local_size
		
		# initialize the process group
		dist.init_process_group("nccl", rank=rank, world_size=world_size)
		

	def cleanup():
		dist.destroy_process_group()
	
	def count_parameters(model):
		return sum(p.numel() for p in model.parameters())


	if configs["is_train"]:

		# setup(rank, world_size)

		model = LitUNet()

		my_auto_wrap_policy = functools.partial(
    		size_based_auto_wrap_policy, min_num_params=20_000_000
			)
		
		# model = FSDP(model,fsdp_auto_wrap_policy=my_auto_wrap_policy)

		prec = torch.float32 #if args.precision == 'float32' else torch.bfloat16

		mixed_precision = MixedPrecision(
				param_dtype=prec,
				reduce_dtype=prec,
				buffer_dtype=prec
			)
		fsdp_strategy = FSDPStrategy(cpu_offload=CPUOffload(True or False),
									sharding_strategy=ShardingStrategy.NO_SHARD, # or GRAD_SHARD or NO_SHARD
									mixed_precision=mixed_precision,
									auto_wrap_policy = my_auto_wrap_policy)
	

		trainer = pl.Trainer(num_nodes=1, strategy = fsdp_strategy, accelerator= 'gpu', devices=2, max_epochs=10)
		x = TrainLoader
		y = ValLoader
		print("Number of parameters with NO_SHARD strategy: ", count_parameters(model))
		trainer.fit(model, train_dataloaders = TrainLoader)  ## x, y  -> trainloader
		


	elif configs["is_test"]:
		PATH_TO_MODEL_CHECKPOINT = configs["path_to_model_checkpoint"]
		model = LitUNet.load_from_checkpoint(PATH_TO_MODEL_CHECKPOINT)
		trainer = pl.Trainer(strategy = 'ddp', accelerator= 'gpu', devices=8)
		x = TestLoader
		trainer.test(model, test_dataloader = TestLoader)

	## test the code ##


	### runs the model, specify number of 

