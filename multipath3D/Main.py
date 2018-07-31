import os
import numpy as np
import time
import sys
import pandas as pd

from model import BrainNet_3D, BrainNet_3D_Inception
from Trainer import Trainer

import torch
from torch.autograd import Variable
import json
# from Inference import Inference

Trainer = Trainer()


nclasses = 5

#--------------------------------------------------------------------------------

def main (nnClassCount=nclasses):
	# "Define Architectures and run one by one"

	nnArchitectureList = [
							{
								'name': 'BrainNet_3D_Inception_layer_in_LAYERS',
								'model' : BrainNet_3D_Inception(),
								'TrainPath': './DataGenerator/oversampled_training_patch_info.csv',
								'ValidPath': './DataGenerator/validation_patch_info.csv',
								'ckpt': '/media/brats/MyPassport/Avinash/Kaminstas_2018/Modified_Kamnistas_Inception_IN_Layer_Model_2018/models/model-m-30072018-153301-BrainNet_3D_Inception_layer_in_LAYERS_loss = 0.9270029878616333_acc = 0.757179012345679_best_loss.pth.tar'
							},

							# {
							# 	'name': 'BrainNet_3D',
							# 	'model' : BrainNet_3D(),
							# 	'TrainPath': './DataGenerator/training_patch_info.csv',
							# 	'ValidPath': './DataGenerator/validation_patch_info.csv',
							# 	'ckpt': None
							# }

						]

	for nnArchitecture in nnArchitectureList:
		runTrain(nnArchitecture=nnArchitecture)



def getDataPaths(path):
 	data = pd.read_csv(path)
 	imgpaths = data['Paths'].as_matrix()
 	return imgpaths

#--------------------------------------------------------------------------------

def runTrain(nnArchitecture = None):

	timestampTime = time.strftime("%H%M%S")
	timestampDate = time.strftime("%d%m%Y")
	timestampLaunch = timestampDate + '-' + timestampTime

	TrainPath = nnArchitecture['TrainPath']
	ValidPath = nnArchitecture['ValidPath']

	#---- Path to the directory with images
	TrainVolPaths = getDataPaths(TrainPath)
	ValidVolPaths = getDataPaths(ValidPath)
		
	np.random.shuffle(TrainVolPaths)	
	np.random.shuffle(ValidVolPaths)

	ValidVolPaths = ValidVolPaths
	TrainVolPaths = TrainVolPaths

	nnClassCount = nclasses

	#---- Training settings: batch size, maximum number of epochs
	trBatchSize = 16
	trMaxEpoch = 50
	learningRate=0.001

	print ('Training NN architecture = ', nnArchitecture['name'])

	info_dict = {
				'batch_size': trBatchSize,
				'architecture':nnArchitecture['name'] ,
				'number of epochs':trMaxEpoch,
				'learningRate':learningRate,
				'train path':TrainPath, 
				'valid_path':ValidPath,
				'number of classes':nclasses,
				'Date-Time':	timestampLaunch
	} 

	os.makedirs('models', exist_ok=True)
	with open('models/config'+str(timestampLaunch)+'.txt','w') as outFile:
		json.dump(info_dict, outFile)
	

	Trainer.train(TrainVolPaths,  ValidVolPaths, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch,learningRate, timestampLaunch, nnArchitecture['ckpt'])



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
