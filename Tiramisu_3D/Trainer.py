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
from torch.autograd import Variable
import torch.nn.functional as F
import torchnet as tnt
import pandas as pd
import random

nclasses = 5
confusion_meter = tnt.meter.ConfusionMeter(nclasses, normalized=True)
from tqdm import tqdm

#--------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor  = y.data if isinstance(y, Variable) else y
    y_tensor  = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims    = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    y_one_hot = y_one_hot.transpose(-1, 1).transpose(-1, 2)#.transpose(-1, 3) 
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot



def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is of the groundtruth, shoud have same size as the input
    """
    # print (target.size())
    target = to_one_hot(target, n_dims=nclasses).to(device)
    # print (target.size(), input.size())

    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 4D Tensor."

    probs = F.softmax(input)

    num   = (probs*target).sum() + 1e-3
    den   = probs.sum() + target.sum() + 1e-3
    dice  = 2.*(num/den)
    return 1. - dice


def normalize(img,mask):
    mean=np.mean(img[mask!=0])
    std=np.std(img[mask!=0])
    return (img-mean)/std

def inner_class_classification(vol, model):
    shape = mask.shape # to exclude batch_size
    final_prediction = np.zeros((shape[0], shape[1], shape[2]))
    x_min, x_max, y_min, y_max, z_min, z_max = 0, (shape[0]-64), 0, (shape[1]-64), 0,(shape[2] - 64)
    with torch.no_grad():
        for x in tqdm(range(x_min, x_max, 32)):
            for y in range(y_min, y_max, 32):
                for z in range(z_min, z_max, 32):
                    temp = vol[:, x:x+64, y:y+64, z:z+64]
                    temp = np.expand_dims(temp, 0)
                    vol_ = torch.from_numpy(temp).to(device).float()
                    pred = torch.exp(model(vol_).detach()).cpu().numpy()[0]
                    final_prediction[x:x+64, y:y+64, z:z+64] = np.argmax(pred, axis = 0)
                    
    return np.uint8(final_prediction)

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

    def train (self, TrainVolPaths, ValidVolPaths, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, timestampLaunch, checkpoint):

        start_epoch=0
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        model = nnArchitecture['model'].to(device)
        # model = torch.nn.DataParallel(model)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER

        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05, weight_decay=1e-5) 
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

        #-------------------- SETTINGS: LOSS
        weigths = np.array([])
        weights   = torch.FloatTensor([0.38398745, 1.48470261, 1.,         1.61940178, 0.2092336])
        loss = torch.nn.CrossEntropyLoss(weight = weights.cuda())

        lossMIN = 100000
        accMax  = 0
        #---- Load checkpoint
        if checkpoint != None:
            saved_parms=torch.load(checkpoint)
            model.load_state_dict(saved_parms['state_dict'])
            optimizer.load_state_dict(saved_parms['optimizer'])
            start_epoch= saved_parms['epochID']
            # lossMIN    = saved_parms['best_loss']
            # accMax     = saved_parms['best_acc']
            print (saved_parms['confusion_matrix'])

        #---- TRAIN THE NETWORK


        sub = pd.DataFrame()

        timestamps = []
        archs = []
        losses = []
        accs = []
        wt_dice_scores=[]
        tc_dice_scores=[]
        et_dice_scores=[]

        for epochID in range (start_epoch, trMaxEpoch):

            #-------------------- SETTINGS: DATASET BUILDERS
            np.random.shuffle(TrainVolPaths)
            np.random.shuffle(ValidVolPaths)
            datasetTrain = custom.ImageFolder(imgs = TrainVolPaths[:40000])
            datasetVal =   custom.ImageFolder(imgs = ValidVolPaths[:2500])

            dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=8, pin_memory=False)
            dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=True, num_workers=8, pin_memory=False)


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

                model_name = '../HGG_models_and_LGG_model/model-m-' + launchTimestamp + "-" + nnArchitecture['name'] + '_loss = ' + str(lossVal) + '_acc = '+str(currAcc) + '_best_loss.pth.tar'
                
                states = {'epochID': epochID + 1,
                            'arch': nnArchitecture['name'],
                            'state_dict': model.state_dict(),
                            'best_acc': currAcc,
                            'confusion_matrix':_cm.conf,
                            'best_loss':lossMIN,
                            'optimizer' : optimizer.state_dict()}

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

                model_name = '../HGG_models_and_LGG_model/model-m-' + launchTimestamp + "-" + nnArchitecture['name'] + '_loss = ' + str(lossVal) + '_acc = '+str(currAcc) + '_best_acc.pth.tar'

                states = {'epochID': epochID + 1,
                            'arch': nnArchitecture['name'],
                            'state_dict': model.state_dict(),
                            'best_acc': accMax,
                            'confusion_matrix':_cm.conf,
                            'best_loss':lossVal,
                            'optimizer' : optimizer.state_dict()}

                torch.save(states, model_name)
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' wt_dice_score='+str(wt_dice_score)+' tc_dice_score='+str(tc_dice_score) +' et_dice_score='+str(et_dice_score) + ' Acc = '+ str(currAcc))


            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + launchTimestamp + '] loss= ' + str(lossVal) + ' wt_dice_score='+str(wt_dice_score)+' tc_dice_score='+str(tc_dice_score) +' et_dice_score='+str(et_dice_score))

            # self.epochTest(model, epochID) # test whole volume

        sub['timestamp'] = timestamps
        sub['archs'] = archs
        sub['loss'] = losses
        sub['WT_dice_score'] = wt_dice_scores
        sub['TC_dice_score'] = tc_dice_scores
        sub['ET_dice_score'] = et_dice_scores

        sub.to_csv('../models/' + nnArchitecture['name'] + '.csv', index=True)

    #--------------------------------------------------------------------------------

    # compute dice
    def get_whole_tumor(self,data):
        return (data>0)*(data<4) 
    
    def get_tumor_core(self,data):
        return np.logical_or(data==1,data==3)

    def get_enhancing_tumor(self,data):
        return data==3


    def get_dice_score(self,prediction, ground_truth):
        masks=(self.get_whole_tumor, self.get_tumor_core, self.get_enhancing_tumor)
        pred = torch.exp(prediction)
        p  = np.uint8(np.argmax(pred.data.cpu().numpy(), axis=1))
        gt = np.uint8(ground_truth.data.cpu().numpy())
        wt,tc,et = [2*np.sum(func(p)*func(gt)) / (np.sum(func(p)) + np.sum(func(gt))+1e-3) for func in masks]
        return wt,tc,et


    #--------------------------------------------------------------------------------
    def epochTrain (self, model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, trBatchSize):

        phase='train'
        with torch.set_grad_enabled(phase == 'train'):
            for batchID, (high_res, seg, weight_map, _) in tqdm(enumerate (dataLoader)):
                
                target = seg.long()
                high_res = high_res.float()
                # weight_map = weight_map.float() / torch.max(weight_map)

                varInputHigh = high_res.to(device)
                varTarget    = target.to(device)
                varMap       = weight_map.to(device)
                # print (varInputHigh.size(), varTarget.size())

                varOutput = model(varInputHigh)
                
                cross_entropy_lossvalue = loss(varOutput, varTarget)

                # assert False
                # cross_entropy_lossvalue = torch.mean(cross_entropy_lossvalue)
                dice_loss_ =  dice_loss(varOutput, varTarget)
                lossvalue  = cross_entropy_lossvalue + dice_loss_
                # lossvalue  = cross_entropy_lossvalue


                # print(lossvalue.size(), varOutput.size(), varMap.size())
                lossvalue = torch.mean(lossvalue)

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
            for i, (high_res, seg, weight_map, _) in enumerate (dataLoader):

                # print (_)
                target = seg.long()
                high_res = high_res.float()
                # weight_map = weight_map.float()/ torch.max(weight_map)

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

                cross_entropy_lossvalue = loss(varOutput, varTarget)

                # assert False
                # cross_entropy_lossvalue = torch.mean(cross_entropy_lossvalue)
                dice_loss_              =  dice_loss(varOutput, varTarget)

                losstensor  =  cross_entropy_lossvalue + dice_loss_
                # losstensor  =  cross_entropy_lossvalue 
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


    # ----------------
    def epochTest(self, model, epochid):
        names     =  os.listdir('/home/uavws/pranjal/MICCAI_BraTS_2018_Data_Training/et')
        model = model.eval()
        
        for name in names:
            root_path = '/home/uavws/pranjal/MICCAI_BraTS_2018_Data_Training/et'+name+'/'
            path = root_path + name +'_'

            brain_mask = np.uint8(nib.load(root_path + 'mask.nii.gz').get_data())
            seg   =  np.uint8(nib.load(path+'seg.nii.gz').get_data())
            seg[(brain_mask != 0) * (seg <= 0)] = 5 #Brain tissue

            flair =  normalize(nib.load(path+'flair.nii.gz').get_data(), brain_mask)
            t1    =  normalize(nib.load(path+'t1.nii.gz').get_data(), brain_mask)
            t1ce  =  normalize(nib.load(path+'t1ce.nii.gz').get_data(), brain_mask)
            t2    =  normalize(nib.load(path+'t2.nii.gz').get_data(), brain_mask)
            affine=  nib.load(path+'seg.nii.gz').affine

            vol = np.zeros((4,)+ t2.shape)
            vol[0, :, :, :] = flair
            vol[1, :, :, :] = t2
            vol[2, :, :, :] = t1
            vol[3, :, :, :] = t1ce

            print (path)
            final_pred     = inner_class_classification(vol, model)

            final_pred[final_pred == 4] = 0
            final_pred[final_pred == 3] = 4

            vol = nib.Nifti1Image(final_pred, affine)
            vol.set_data_dtype(np.uint8)
            nib.save(vol, root_path + epochid +'.nii.gz')

        pass