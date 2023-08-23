#!/usr/bin/env python

import argparse
from argparse import Namespace, ArgumentParser
import os





import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from data_utils import SimuData, test_prediction, analysis
from fsdp_utilities import *
from fsdp_utilities import setup_environment
import sys
#sys.path.insert(0, '/mnt/home/siyuh/Project/Recon/Unet/')
sys.path.insert(0, '/lcrc/project/cosmo_ai/dongx/PM-128-redshift/0.3-0.4/')
sys.path.insert(0, '/lcrc/project/cosmo_ai/dongx/PM-128-redshift/0.3-0.4/ML-Recon/Unet')
from uNet import BasicBlock, Lpt2NbodyNet


rank = int(os.environ.get('PMI_RANK', '0'))
zero = rank == 0

def count_parameters(model):
		return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":


	args = get_args()
	args = setup_environment(args)
	args = init_distributed(args)
	model = setup_model(args)
	# print(model)

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
	# os.environ["CUDA_CACHE_PATH"] = args.cuda_cache_path
	
	with open(args.config_file_path) as f:
		configs = json.load(f)

	print("Number of parameters with FULL_SHARD strategy: ", count_parameters(model))
	# net = Lpt2NbodyNet(BasicBlock)

	# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
	
	# net = nn.DataParallel(net)
	# net = net.to(device)

#	net.cuda()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=configs["net_params"]['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=configs["net_params"]['reg'])

	base_data_path = configs["base_data_path"]
	output_path = configs["output_path"]

	if configs["is_train"]:
		TrainLoader, ValLoader = get_dataloaders(args)

		# TrainSet = SimuData(base_data_path,
		# 			configs['train']['data_partition']['label'],
		# 			#configs['train']['data_partition']['hIndex'],
		# 			configs['train']['data_partition']['aug'])
		# ValSet = SimuData(base_data_path,
		# 			configs['val']['data_partition']['label'],
		# 			#configs['val']['data_partition']['lIndex'],
		# 			#configs['val']['data_partition']['hIndex'],
		# 			configs['val']['data_partition']['aug'])
		# TrainLoader = DataLoader(TrainSet,
		# 			batch_size=configs['train']['batch_size'],
		# 			shuffle=True,
		# 			num_workers=configs['val']['num_workers'])
		# ValLoader   = DataLoader(ValSet,
		# 			batch_size=configs['val']['batch_size'],
		# 			shuffle=False,
		# 			num_workers=configs['val']['num_workers'])
		
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
	
	if(configs["is_train"]):
		print('training begins')
		print('how many GPUs?')
		print(torch.cuda.device_count())
		for _ in range(configs['train']['num_epoches']):
			print('epoch:')
			print(_)
			for t, data in enumerate(TrainLoader, 0):
				#print("t="+str(t)+" and data="+str(data))
				#print(np.array(data).shape)
				#print("t="+str(t))
				
				start_time = time.time()
				model.train()
				optimizer.zero_grad()
				#if(t==0):
					#np.save('/lcrc/project/cosmo_ai/dongx/UNET-64-new/debug_output/data0.npy',data[0])
					#np.save('/lcrc/project/cosmo_ai/dongx/UNET-64-new/debug_output/data1.npy',data[1])


				#NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
				NetInput = torch.autograd.Variable(data[0],requires_grad=False) 
				#.to(device)
				if(zero):
					print('NetInput shape:')
					print(NetInput.shape)
				# Y_pred = model(NetInput)
				# print(Y_pred)
				# loss = criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda())
				target = torch.autograd.Variable(data[1],requires_grad=False)
				loss = model(NetInput, target) # criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False)) #.to(device)
				loss_train.append(loss.item())
				loss.backward()
				optimizer.step()
				if zero :
					print (iterTime,loss.item())
				np.savetxt(output_path+'trainLoss.txt',loss_train)
				if(iterTime!=0 and iterTime%eval_frequency ==0):
					model.eval()
					start_time = time.time()
					_loss=0
					for t_val, data in enumerate(ValLoader,0):
						with torch.no_grad():
							# NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
							NetInput = torch.autograd.Variable(data[0],requires_grad=False)#.to(device)
							loss_new = model(NetInput)
							# _loss += criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda()).item()
							_loss += loss_new #criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False)).item()
					loss_val.append(_loss/t_val)
					np.savetxt(output_path+'valLoss.txt',loss_val)
					if zero:
						print ('valid: ' + str(_loss/t_val))
					if( _loss/t_val < best_validation_accuracy):
						torch.save(net,output_path+'BestModel.pt')
					if(iterTime%500==0 and iterTime!=0):
						torch.save(net,output_path+'BestModel_8batchsize_'+str(iterTime)+'.pt')
						print('best model saved at: ' )
						print(iterTime)
				iterTime+=1						
		save_restart_checkpoint('./checkpoints',
                                    'FINAL',
                                    model,
                                    optimizer,
                                    args,
                                    _)
		

		if zero :
			print('\n \n Training Finally Complete!! Hooray! \n \n')
	if(configs["is_test"]):
		test_prediction(output_path,configs["test"]["model"],TestLoader)

	if(configs["is_analysis"]):
		net = torch.load(output_path+'BestModel.pt')
		net.cuda()
		net.eval()
		#np.save('layer1_weight.npy',net.layer1[0].conv1.weight.data.cpu().numpy())
		#analysis(output_path,"BestModel.pt",configs["analysis"]["size"],configs["analysis"]["A"],configs["analysis"]["phi"],configs["analysis"]["k"])
