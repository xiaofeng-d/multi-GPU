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
from torch.utils.data import DataLoader
import time
from data_utils import SimuData, test_prediction, analysis
import sys


import torch
import torch.nn as nn
from periodic_padding import periodic_padding_3d
from data_utils import crop_tensor
import numpy as np


def test_prediction(path,model,TestLoader):
	net = torch.load(model)
	net.cuda()
	net.eval()

	for t, data in enumerate(TestLoader, 0):
		print (t)
		NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
		#np.save('/home/dongx/UNET/debug_output/data0'+str(t)+'.npy',data[0])
		Y_pred = net(NetInput)
		#np.save('/home/dongx/UNET/debug_output/Y_pred'+str(t)+'.npy',np.squeeze(Y_pred.data.cpu().numpy()))
		np.save(path+'PM_0405test_'+str(t)+'.npy',np.concatenate((np.squeeze(Y_pred.data.cpu().numpy()),np.squeeze(data[1].numpy()), np.squeeze(data[0].numpy()) ),axis=0))
		

class SimuData(Dataset):  ## XFD Version --- with the correct splitting of test and training
	def __init__(self,base_path,label,aug):
		self.datafiles = []
		#for x in np.arange(lIndex,hIndex,1000):
			#y = [base_path+str(x)+'_'+str(i)+'.npy' for i in range(1000)]
		y = [base_path + 'lagr_' + str(i) + '_'+ label + '.npy' for i in range(2)]
		print(y)
		self.datafiles+=y
		self.aug=aug
		self.label=label

	def __getitem__(self, index):
		return get_mini_batch(self.datafiles, index, self.aug)

	def __len__(self):
		if(self.label=="val"):
			return 199  ## needs thinking 
		if(self.label=="train"):
			return 399
		if(self.label=="test"):
			return 200



def conv3x3(inplane,outplane, stride=1,padding=0):
	return nn.Conv3d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)

class BasicBlock(nn.Module):
	def __init__(self,inplane,outplane,stride = 1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplane,outplane,padding=0,stride=stride)
		self.bn1 = nn.BatchNorm3d(outplane)
		self.relu = nn.ReLU(inplace=True)

	def forward(self,x):
		x = periodic_padding_3d(x,pad=(1,1,1,1,1,1))
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		return out

class Lpt2NbodyNet(nn.Module):
	def __init__(self, block):
		super(Lpt2NbodyNet,self).__init__()
		self.layer1 = self._make_layer(block, 3, 64, blocks=2,stride=1)
		self.layer2 = self._make_layer(block,64,128, blocks=1,stride=2)
		self.layer3 = self._make_layer(block,128,128,blocks=2,stride=1)
		self.layer4 = self._make_layer(block,128,256,blocks=1,stride=2)
		self.layer5 = self._make_layer(block,256,256,blocks=2,stride=1)
		self.deconv1 = nn.ConvTranspose3d(256,128,3,stride=2,padding=0)
		self.deconv_batchnorm1 = nn.BatchNorm3d(num_features = 128,momentum=0.1)
		self.layer6 = self._make_layer(block,256,128,blocks=2,stride=1)
		self.deconv2 = nn.ConvTranspose3d(128,64,3,stride=2,padding=0)
		self.deconv_batchnorm2 = nn.BatchNorm3d(num_features = 64,momentum=0.1)
		self.layer7 = self._make_layer(block,128,64,blocks=2,stride=1)
		self.deconv4 = nn.ConvTranspose3d(64,3,1,stride=1,padding=0)



	def _make_layer(self,block,inplanes,outplanes,blocks,stride=1):
		layers = []
		for i in range(0,blocks):
			layers.append(block(inplanes,outplanes,stride=stride))
			inplanes = outplanes
		return nn.Sequential(*layers)

	def forward(self,x):
		x1 = self.layer1(x)
		x  = self.layer2(x1)
		x2 = self.layer3(x)
		x  = self.layer4(x2)
		x  = self.layer5(x)
		x  = periodic_padding_3d(x,pad=(0,1,0,1,0,1))
		x  = nn.functional.relu(self.deconv_batchnorm1(crop_tensor(self.deconv1(x))),inplace=True)
		x  = torch.cat((x,x2),dim=1)
		x  = self.layer6(x)
		x  = periodic_padding_3d(x,pad=(0,1,0,1,0,1))
		x  = nn.functional.relu(self.deconv_batchnorm2(crop_tensor(self.deconv2(x))),inplace=True)
		x  = torch.cat((x,x1),dim=1)
		x  = self.layer7(x)
		x  = self.deconv4(x)

		return x

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()
	with open(args.config_file_path) as f:
		configs = json.load(f)

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
	os.environ["CUDA_CACHE_PATH"] = args.cuda_cache_path

	net = Lpt2NbodyNet(BasicBlock)
	device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
	net = nn.DataParallel(net)
	net = net.to(device)
#	net.cuda()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(),lr=configs["net_params"]['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=configs["net_params"]['reg'])

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
				net.train()
				optimizer.zero_grad()
				#if(t==0):
					#np.save('/lcrc/project/cosmo_ai/dongx/UNET-64-new/debug_output/data0.npy',data[0])
					#np.save('/lcrc/project/cosmo_ai/dongx/UNET-64-new/debug_output/data1.npy',data[1])


				#NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
				NetInput = torch.autograd.Variable(data[0],requires_grad=False).to(device)
				Y_pred = net(NetInput)
				# loss = criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda())
				loss = criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).to(device))
				loss_train.append(loss.item())
				loss.backward()
				optimizer.step()
				print (iterTime,loss.item())
				np.savetxt(output_path+'trainLoss.txt',loss_train)
				if(iterTime!=0 and iterTime%eval_frequency ==0):
					net.eval()
					start_time = time.time()
					_loss=0
					for t_val, data in enumerate(ValLoader,0):
						with torch.no_grad():
							# NetInput = torch.autograd.Variable(data[0],requires_grad=False).cuda()
							NetInput = torch.autograd.Variable(data[0],requires_grad=False).to(device)
							Y_pred = net(NetInput)
							# _loss += criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).cuda()).item()
							_loss += criterion(Y_pred, torch.autograd.Variable(data[1],requires_grad=False).to(device)).item()
					loss_val.append(_loss/t_val)
					np.savetxt(output_path+'valLoss.txt',loss_val)
					print ('valid: ' + str(_loss/t_val))
					if( _loss/t_val < best_validation_accuracy):
						torch.save(net,output_path+'BestModel.pt')
					if(iterTime%500==0 and iterTime!=0):
						torch.save(net,output_path+'BestModel_8batchsize_'+str(iterTime)+'.pt')
						print('best model saved at: ' )
						print(iterTime)

				iterTime+=1
	if(configs["is_test"]):
		test_prediction(output_path,configs["test"]["model"],TestLoader)

	if(configs["is_analysis"]):
		net = torch.load(output_path+'BestModel.pt')
		net.cuda()
		net.eval()
		#np.save('layer1_weight.npy',net.layer1[0].conv1.weight.data.cpu().numpy())
		#analysis(output_path,"BestModel.pt",configs["analysis"]["size"],configs["analysis"]["A"],configs["analysis"]["phi"],configs["analysis"]["k"])
