import os
import numpy as np
import time
import sys
import pandas as pd

from model import FCDenseNet103, FCDenseNet57, FCDenseNet67
from Trainer import Trainer

import torch
from torch.autograd import Variable
import json

# from Inference import Inference

Trainer = Trainer()


nclasses = 4

#--------------------------------------------------------------------------------

def main (nnClassCount=nclasses):
	# "Define Architectures and run one by one"

	nnArchitectureList = [
							{
								'name': 'tramisu_2D_FC103',
								'model' : FCDenseNet103(n_classes = nclasses),
								'TrainPath': '/media/brats/MyPassport/Avinash/Kaminstas_2018/MICCAI_BraTS_2018_Data_Training/hdf5_data_0_255/train_set',
								'ValidPath': '/media/brats/MyPassport/Avinash/Kaminstas_2018/MICCAI_BraTS_2018_Data_Training/hdf5_data_0_255/valid_set',
								'ckpt' : None
							}

						]

	for nnArchitecture in nnArchitectureList:
		runTrain(nnArchitecture=nnArchitecture)





#--------------------------------------------------------------------------------

def runTrain(nnArchitecture = None):

	timestampTime = time.strftime("%H%M%S")
	timestampDate = time.strftime("%d%m%Y")
	timestampLaunch = timestampDate + '-' + timestampTime

	TrainPath = nnArchitecture['TrainPath']
	ValidPath = nnArchitecture['ValidPath']

	nnClassCount = nclasses

	#---- Training settings: batch size, maximum number of epochs
	trBatchSize = 4
	trMaxEpoch = 30

	print ('Training NN architecture = ', nnArchitecture['name'])
	info_dict = {
				'batch_size': trBatchSize,
				'architecture':nnArchitecture['name'] ,
				'number of epochs':trMaxEpoch,
				'train path':TrainPath, 
				'valid_path':ValidPath,
				'number of classes':nclasses,
				'Date-Time':	timestampLaunch
				} 

	root = '../models_103_4_sequences_mri_training_stats_normalization'
	if not os.path.exists(root):
		os.mkdir(root)

	with open(root+'/config.txt','w') as outFile:
		json.dump(info_dict, outFile)
	

	Trainer.train(TrainPath,  ValidPath, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, timestampLaunch, nnArchitecture['ckpt'],root)


#--------------------------------------------------------------------------------

def runTest():

	Path = '../processed_data/train_test_split.csv'
	TestVolPaths = getDataPaths(Path, 'Testing')
	nnClassCount = nclasses

	trBatchSize = 1
	imgtransResize = 64
	imgtransCrop = 64

	pathsModel = ['../models/densenet3D.csv']

	timestampLaunch = ''

	# nnArchitecture = DenseNet121(nnClassCount, nnIsTrained)
	print ('Testing the trained model')
	Tester.test(TestVolPaths, pathsModel, nnClassCount, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
#--------------------------------------------------------------------------------

if __name__ == '__main__':
	main()
	# runTest()
