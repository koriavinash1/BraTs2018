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


nclasses = 5

#--------------------------------------------------------------------------------

def main (nnClassCount=nclasses):
    # "Define Architectures and run one by one"

    nnArchitectureList = [
                            {
                                'name': 'tramisu_3D_FC103',
                                'model' : FCDenseNet103(n_classes = nclasses),
                                'TrainPath': './training_patch_info.csv',
                                'ValidPath': './validation_patch_info.csv',
                                'ckpt' : '/home/uavws/pranjal/HGG_models/model-m-28072018-090622-tramisu_3D_FC103_loss = 0.3187295208809686_acc = 0.9395081489562989_best_acc.pth.tar'
                            }

                        ]

    for nnArchitecture in nnArchitectureList:
        runTrain(nnArchitecture=nnArchitecture)



def getDataPaths(path):
    data = pd.read_csv(path)
    imgpaths = [str_ for str_ in data['Paths'].as_matrix()] # if str_.__contains__('HGG')]
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

    TrainVolPaths = TrainVolPaths
    ValidVolPaths = ValidVolPaths

    nnClassCount = nclasses

    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 3
    trMaxEpoch = 30

    print ('Training NN architecture = ', nnArchitecture['name'])
    info_dict = {
                'batch_size': trBatchSize,
                'architecture':nnArchitecture['name'] ,
                'number of epochs':trMaxEpoch,
                'train path':TrainPath, 
                'valid_path':ValidPath,
                'number of classes':nclasses,
                'Date-Time':    timestampLaunch,
                'Network' : 'FCDenseNet(\
                            in_channels=4, down_blocks=(4,5,7,10),\
                            up_blocks=(10,7,5,4), bottleneck_layers=7,\
                            growth_rate=8, out_chans_first_conv=24, n_classes=n_classes)'
    } 

    os.makedirs('../HGG_models_and_LGG_model/', exist_ok=True)
    with open('../HGG_models_and_LGG_model/config.txt','w') as outFile:
        json.dump(info_dict, outFile)
    

    Trainer.train(TrainVolPaths,  ValidVolPaths, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, timestampLaunch, nnArchitecture['ckpt'])


#--------------------------------------------------------------------------------

def runTest():

    Path = '../processed_data/train_test_split.csv'
    TestVolPaths = getDataPaths(Path, 'Testing')
    nnClassCount = nclasses

    trBatchSize = 1
    imgtransResize = 64
    imgtransCrop = 64

    pathsModel = ['../HGG_models_and_LGG_model/densenet3D.csv']

    timestampLaunch = ''

    # nnArchitecture = DenseNet121(nnClassCount, nnIsTrained)
    print ('Testing the trained model')
    Tester.test(TestVolPaths, pathsModel, nnClassCount, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
    # runTest()
