import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
from torch.utils.data import DataLoader

from sklearn.metrics.ranking import roc_auc_score

import dl as custom
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import torchnet as tnt
import pandas as pd
import random

nclasses=4
confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)
from tqdm import tqdm

#--------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer ():
	#---- Train the densenet network
	#---- TrainVolPaths - path to the directory that contains images
	#---- TrainLabels - path to the file that contains image paths and label pairs (training set)
	#---- ValidVolPaths - path to the directory that contains images
	#---- ValidLabels - path to the file that contains image paths and label pairs (training set)
	#---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
	#---- nnClassCount - number of output classes
	#---- trBatchSize - batch size
	#---- trMaxEpoch - number of epochs
	#---- transResize - size of the image to scale down to (not used in current implementation)
	#---- transCrop - size of the cropped image
	#---- launchTimestamp - date/time, used to assign unique name for the checkpoint file

	#---- TODO:
	#---- checkpoint - if not None loads the model and continues training

	def train (self, TrainPath, ValidPath, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, timestampLaunch, checkpoint,root):

		start_epoch=0
		#-------------------- SETTINGS: NETWORK ARCHITECTURE
		model = nnArchitecture['model'].to(device)

		# model = torch.nn.DataParallel(model)

		#-------------------- SETTINGS: OPTIMIZER & SCHEDULER

		optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05, weight_decay=1e-5)
		scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

		# normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		normalize = transforms.Normalize([0.4465654,  0.39885548, 0.382155,   0.6104794],[0.1427262,  0.14297858, 0.09494335, 0.13883562])
		
		transformList = []
		transformList.append(transforms.ToTensor())
		transformList.append(normalize)
		transformSequence=transforms.Compose(transformList)

		
		#-------------------- SETTINGS: DATASET BUILDERS
		datasetTrain = custom.ImageFolder(TrainPath,transformSequence)
		datasetVal =   custom.ImageFolder(ValidPath,transformSequence)

		dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=4, pin_memory=False)
		dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=4, pin_memory=False)


		#-------------------- SETTINGS: LOSS
		loss = torch.nn.CrossEntropyLoss(reduce = False) # for pixel wise crossentropy for weight maps....
		
		# weights=torch.FloatTensor([0.00887307, 2.64364268, 0.69395684, 1.78894656]).to(device)
		# loss = torch.nn.CrossEntropyLoss(weight=weights)
		

		lossMIN = 100000
		accMax  =0

		timestamps = []
		archs = []
		losses = []
		accs = []
		wt_dice_scores=[]
		tc_dice_scores=[]
		et_dice_scores=[]

		#---- Load checkpoint
		if checkpoint != None:
			saved_parms=torch.load(checkpoint)
			model.load_state_dict(saved_parms['state_dict'])
			optimizer.load_state_dict(saved_parms['optimizer'])
			start_epoch= saved_parms['epochID']
			lossMIN    = saved_parms['best_loss']
			accMax     =saved_parms['best_acc']
			timestamps =saved_parms['timestamps']
			archs =saved_parms['archs']
			loss  =saved_parms['loss']
			wt_dice_scores=saved_parms['WT_dice_score']
			tc_dice_scores=saved_parms['TC_dice_score']
			et_dice_scores=saved_parms['ET_dice_score']
			print (saved_parms['confusion_matrix'])


		#---- TRAIN THE NETWORK
		sub = pd.DataFrame()


		for epochID in range (start_epoch, trMaxEpoch):

			timestampTime = time.strftime("%H%M%S")
			timestampDate = time.strftime("%d%m%Y")
			timestampSTART = timestampDate + '-' + timestampTime

			print (str(epochID)+"/" + str(trMaxEpoch) + "---")
			self.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)
			lossVal, losstensor, wt_dice_score, tc_dice_score, et_dice_score, _cm = self.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, trBatchSize)

			currAcc = float(np.sum(np.eye(nclasses)*_cm.conf))/np.sum(_cm.conf)
			print (_cm.conf)

			timestampTime = time.strftime("%H%M%S")
			timestampDate = time.strftime("%d%m%Y")
			launchTimestamp = timestampDate + '-' + timestampTime

			scheduler.step(losstensor.item())

			if lossVal < lossMIN:
				lossMIN = lossVal

				timestamps.append(launchTimestamp)
				archs.append(nnArchitecture['name'])
				losses.append(lossVal)
				accs.append(currAcc)
				wt_dice_scores.append(wt_dice_score)
				tc_dice_scores.append(tc_dice_score)
				et_dice_scores.append(et_dice_score)

				# model_name = root+'/model-m-' + launchTimestamp + "-" + nnArchitecture['name'] + '_loss = ' + str(lossVal) + '_acc = '+str(currAcc) + '_best_loss.pth.tar'
				model_name   = root + '/best_loss_model.pth.tar'

				states = {'epochID': epochID + 1,
							'arch': nnArchitecture['name'],
							'state_dict': model.state_dict(),
							'best_acc': currAcc,
							'confusion_matrix':_cm.conf,
							'best_loss':lossMIN,
							'optimizer' : optimizer.state_dict(),
							'timestamps':timestamps,
							'archs':archs,
							'loss':loss,
							'WT_dice_score':wt_dice_scores,
							'TC_dice_score':tc_dice_scores,
							'ET_dice_score':et_dice_scores }

				torch.save(states, model_name)
				print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' wt_dice_score='+str(wt_dice_score)+' tc_dice_score='+str(tc_dice_score) +' et_dice_score='+str(et_dice_score))

			elif currAcc > accMax:
				accMax  = currAcc

				timestamps.append(launchTimestamp)
				archs.append(nnArchitecture['name'])
				losses.append(lossVal)
				accs.append(accMax)
				wt_dice_scores.append(wt_dice_score)
				tc_dice_scores.append(tc_dice_score)
				et_dice_scores.append(et_dice_score)

				# model_name = root+'/model-m-' + launchTimestamp + "-" + nnArchitecture['name'] + '_loss = ' + str(lossVal) + '_acc = '+str(currAcc) + '_best_acc.pth.tar'
				model_name   = root + '/best_acc_model.pth.tar'

				states = {'epochID': epochID + 1,
							'arch': nnArchitecture['name'],
							'state_dict': model.state_dict(),
							'best_acc': accMax,
							'confusion_matrix':_cm.conf,
							'best_loss':lossVal,
							'optimizer' : optimizer.state_dict(),
							'timestamps':timestamps,
							'archs':archs,
							'loss':loss,
							'WT_dice_score':wt_dice_scores,
							'TC_dice_score':tc_dice_scores,
							'ET_dice_score':et_dice_scores
							}

				torch.save(states, model_name)
				print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' wt_dice_score='+str(wt_dice_score)+' tc_dice_score='+str(tc_dice_score) +' et_dice_score='+str(et_dice_score) + ' Acc = '+ str(currAcc))


			else:
				print ('Epoch [' + str(epochID + 1) + '] [----] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' wt_dice_score='+str(wt_dice_score)+' tc_dice_score='+str(tc_dice_score) +' et_dice_score='+str(et_dice_score))

		sub['timestamp'] = timestamps
		sub['archs'] = archs
		sub['loss']  = losses
		sub['WT_dice_score'] = wt_dice_scores
		sub['TC_dice_score'] = tc_dice_scores
		sub['ET_dice_score'] = et_dice_scores

		sub.to_csv(root + '/' + nnArchitecture['name'] + '.csv', index=True)

	#--------------------------------------------------------------------------------

	# compute dice
	def get_whole_tumor(self,data):
		return (data>0)*(data<4)

	def get_tumor_core(self,data):
		return np.logical_or(data==1,data==3)

	def get_enhancing_tumor(self,data):
		return data==3


	def get_dice_score(self,prediction,ground_truth):
		masks=(self.get_whole_tumor, self.get_tumor_core, self.get_enhancing_tumor)
		pred=torch.exp(prediction)
		p=np.uint8(np.argmax(pred.data.cpu().numpy(), axis=1))
		gt=np.uint8(ground_truth.data.cpu().numpy())
		wt,tc,et=[2*np.sum(func(p)*func(gt)) / (np.sum(func(p)) + np.sum(func(gt))+1e-3) for func in masks]
		return wt,tc,et


	#--------------------------------------------------------------------------------
	def epochTrain (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):

		phase='train'
		with torch.set_grad_enabled(phase == 'train'):
			for high_res, seg, weight_map,_ in tqdm(dataLoader):
				# print (high_res.size(), seg.size())
				target = seg.long()
				high_res = high_res.float()
				weight_map = weight_map.float()

				varInputHigh = high_res.to(device)
				varTarget    = target.to(device)
				varMap       = weight_map.to(device)
				# if torch.isnan(torch.max(varInputHigh)) or torch.isnan(torch.min(varTarget)): continue
				# print (torch.max(varInputHigh), torch.min(varTarget))

				varOutput = model(varInputHigh)
				lossvalue = loss(varOutput, varTarget)*varMap

				# assert False
				lossvalue = torch.mean(lossvalue)
				# lossvalue= loss(varOutput, varTarget)


				optimizer.zero_grad()
				lossvalue.backward()
				optimizer.step()

	#--------------------------------------------------------------------------------
	def epochVal (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):

		model.eval ()

		lossVal = 0
		lossValNorm = 0

		losstensorMean = 0
		confusion_meter.reset()

		wt_dice_score, tc_dice_score, et_dice_score = 0.0, 0.0, 0.0
		with torch.no_grad():
			for high_res, seg, weight_map,_ in tqdm(dataLoader):

				# print (_)
				target   = seg.long()
				high_res = high_res.float()
				weight_map = weight_map.float()

				varInputHigh = high_res.to(device)
				varTarget    = target.to(device)
				varMap       = weight_map.to(device)
				# print (varInputHigh.size(), varTarget.size())

				varOutput = model(varInputHigh)
				_, preds = torch.max(varOutput,1)

				wt_, tc_, et_ = self.get_dice_score(varOutput, varTarget)
				wt_dice_score += wt_
				tc_dice_score += tc_
				et_dice_score += et_

				losstensor = loss(varOutput, varTarget)*varMap
				losstensor = torch.mean(losstensor)

				# losstensor = loss(varOutput, varTarget)
				# print varOutput, varTarget
				losstensorMean += losstensor
				confusion_meter.add(preds.data.view(-1), varTarget.data.view(-1))
				lossVal += losstensor.item()
				del losstensor,_,preds
				del varOutput, varTarget, varInputHigh
				lossValNorm += 1

			wt_dice_score, tc_dice_score, et_dice_score = wt_dice_score/lossValNorm, tc_dice_score/lossValNorm, et_dice_score/lossValNorm
			outLoss = lossVal / lossValNorm
			losstensorMean = losstensorMean / lossValNorm

		return outLoss, losstensorMean, wt_dice_score, tc_dice_score, et_dice_score, confusion_meter
