#!/usr/bin/env python

import argparse

def get_parser():
	parser = argparse.ArgumentParser(description="U-Net for ZA -> Nbody net")
	parser.add_argument("cuda_visible_devices", type=str)
	parser.add_argument("--cuda_cache_path", type=str,
		default="/home/scratch/siyuh/nv_ComputeCache")  # to fix issue with auton lab gpu
	parser.add_argument("-c", "--config_file_path", type=str, default='')
	return parser

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from torch.utils.data import DataLoader
import time
from data_utils import SimuData, test_prediction, analysis
import sys
#sys.path.insert(0, '/mnt/home/siyuh/Project/Recon/Unet/')
sys.path.insert(0, '/lcrc/project/cosmo_ai/dongx/Multi-GPU/0.3-0.4-multiGPU/')
sys.path.insert(0, '/lcrc/project/cosmo_ai/dongx/PM-128-redshift/0.3-0.4/ML-Recon/Unet')
from uNet import BasicBlock, Lpt2NbodyNet

if __name__ == "__main__":

	parser = get_parser()
	args = parser.parse_args()
	with open(args.config_file_path) as f:
		configs = json.load(f)

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
	os.environ["CUDA_CACHE_PATH"] = args.cuda_cache_path

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

			np.save(save_path, np.concatenate((np.squeeze(Y_pred_np),np.squeeze(y.cpu().detach().numpy()), np.squeeze(x.cpu().detach().numpy())),axis=0))
			# np.save(save_path, Y_pred_np)

			self.log('test_loss', loss, on_epoch = True)
			
			return loss
		
		
		def validation_step(self, val_batch, batch_idx):
			x, y = val_batch
			# x, y = val_batch


	## set up traning and validation data  
	print('how many GPUs?')
	print(torch.cuda.device_count())
	
	
	if configs["is_train"]:
		model = LitUNet()
		trainer = pl.Trainer(strategy = 'ddp', accelerator= 'gpu', devices=8)
		x = TrainLoader
		y = ValLoader
		trainer.fit(model, train_dataloaders = TrainLoader)  ## x, y  -> trainloader

	elif configs["is_test"]:
		PATH_TO_MODEL_CHECKPOINT = configs["path_to_model_checkpoint"]
		model = LitUNet.load_from_checkpoint(PATH_TO_MODEL_CHECKPOINT)
		trainer = pl.Trainer(strategy = 'ddp', accelerator= 'gpu', devices=1)
		x = TestLoader
		trainer.test(model, dataloaders = TestLoader)

	## test the code ##


	### runs the model, specify number of 

	

#	net.cuda()\
	# criterion = nn.MSELoss()
	# optimizer = torch.optim.Adam(net.parameters(),lr=configs["net_params"]['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=configs["net_params"]['reg'])

	# if(configs["is_train"]):
	# 	print('training begins')
	# 	print('how many GPUs?')
	# 	print(torch.cuda.device_count())
	# 	for _ in range(configs['train']['num_epoches']):
	# 		print('epoch:')
	# 		print(_)
	# 		for t, data in enumerate(TrainLoader, 0):
	# 			#print("t="+str(t)+" and data="+str(data))
	# 			#print(np.array(data).shape)
	# 			#print("t="+str(t))
				
	# 			start_time = time.time()
	# 			net.train()
	# 			optimizer.zero_grad()
	# 			#if(t==0):
	# 				#np.save('/lcrc/project/cosmo_ai/dongx/UNET-64-new/debug_output/data0.npy',data[0])
	# 				#np.save('/lcrc/project/cosmo_ai/dongx/UNET-64-new/debug_output/data1.npy',data[1])


	# 			#NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
	# 			NetInput = torch.autograd.Variable(data[0],requires_grad=False).to(device)
	# 			Y_pred = net(NetInput)
	# 			# loss = criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda())
	# 			loss = criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).to(device))
	# 			loss_train.append(loss.item())
	# 			loss.backward()
	# 			optimizer.step()
	# 			print (iterTime,loss.item())
	# 			np.savetxt(output_path+'trainLoss.txt',loss_train)
	# 			if(iterTime!=0 and iterTime%eval_frequency ==0):
	# 				net.eval()
	# 				start_time = time.time()
	# 				_loss=0
	# 				for t_val, data in enumerate(ValLoader,0):
	# 					with torch.no_grad():
	# 						# NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
	# 						NetInput = torch.autograd.Variable(data[0],requires_grad=False).to(device)
	# 						Y_pred = net(NetInput)
	# 						# _loss += criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda()).item()
	# 						_loss += criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).to(device)).item()
	# 				loss_val.append(_loss/t_val)
	# 				np.savetxt(output_path+'valLoss.txt',loss_val)
	# 				print ('valid: ' + str(_loss/t_val))
	# 				if( _loss/t_val < best_validation_accuracy):
	# 					torch.save(net,output_path+'BestModel.pt')
	# 				if(iterTime%500==0 and iterTime!=0):
	# 					torch.save(net,output_path+'BestModel_8batchsize_'+str(iterTime)+'.pt')
	# 					print('best model saved at: ' )
	# 					print(iterTime)

	# 			iterTime+=1
	# if(configs["is_test"]):
	# 	test_prediction(output_path,configs["test"]["model"],TestLoader)

	# if(configs["is_analysis"]):
	# 	net = torch.load(output_path+'BestModel.pt')
	# 	net.cuda()
	# 	net.eval()
	# 	#np.save('layer1_weight.npy',net.layer1[0].conv1.weight.data.cpu().numpy())
	# 	#analysis(output_path,"BestModel.pt",configs["analysis"]["size"],configs["analysis"]["A"],configs["analysis"]["phi"],configs["analysis"]["k"])
