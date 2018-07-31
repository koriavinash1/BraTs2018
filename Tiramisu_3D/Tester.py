
# coding: utf-8

# In[6]:


import torch
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.measurements import label
from skimage.morphology import erosion, dilation
import nibabel as nib
from torch.autograd import Variable
from skimage.transform import resize
from tqdm import tqdm
import os
from model import FCDenseNet57, FCDenseNet103
from scipy.ndimage import uniform_filter, maximum_filter
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

nclasses = 5
net = FCDenseNet103(5)


# In[7]:


# ckpt = torch.load('/media/brats/Kori/Kaminstas_2018/models/model-m-15062018-233418-BrainNet_3D_Inception_loss = 0.7610744878227615_acc = 0.7903060229355852_best_acc.pth.tar')
ckpt = torch.load('/home/uavws/pranjal/HGG_models/model-m-27072018-085701-tramisu_3D_FC103_loss = 0.2847366505648872_acc = 0.9327735084533691_best_loss.pth.tar')
net.load_state_dict(ckpt['state_dict'])


# In[5]:


net.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[6]:


def bbox(vol):
    tumor = np.where(vol>0)

    x_min = np.min(tumor[0])
    y_min = np.min(tumor[1])
    z_min = np.min(tumor[2])

    x_max = np.max(tumor[0])
    y_max = np.max(tumor[1])
    z_max = np.max(tumor[2])
    
    return x_min, x_max, y_min, y_max, z_min, z_max        


def perform_postprocessing(vol, thresh = 0.50, verbose=False):
    label_vol, label_masks = label(vol)
    volumes = []
    for i in range(label_masks):
        x, y, z = np.where(label_vol == i+1)
        v = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y)) * (np.max(z) - np.min(z))
        volumes.append(v)
        
    if verbose:	print ("Max. Volume: {}".format(np.max(volumes)) + " Min. Volumes: {}".format(np.min(volumes)))

    for i in range(len(volumes)):
        if volumes[i] < np.max(volumes) * thresh:
            label_vol[label_vol == i+1] = 0

    label_vol[label_vol > 0] = 1
    return label_vol * vol

def normalize(img,mask):
    mean=np.mean(img[mask!=0])
    std=np.std(img[mask!=0])
    return (img-mean)/std

def get_localization(t1, t1ce, t2, flair):
    mask_rcnn_pred = 0
    return mask_rcnn_pred\

def crf(vol, prob):
    # prob shape: classes, width, height, depth
    # vol shape: width, height, depth
    assert len(prob.shape) == 4
    assert len(vol.shape) == 4
    return_ = np.empty((240,240,155))
    d = dcrf.DenseCRF(np.prod((240,240,155)), prob.shape[0])
    # get unary potentials (neg log probability)
    U = unary_from_softmax(prob)
    d.setUnaryEnergy(U)
    feats = create_pairwise_gaussian(sdims=(1., 1., 1.), shape=(240,240,155))
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    print (np.max(Q), np.min(Q))
    return_ = np.argmax(Q, axis=0).reshape(240, 240, 155)
    return return_


def inner_class_classification(vol, model):
    shape = mask.shape # to exclude batch_size
    final_prediction = np.empty((5, shape[0], shape[1], shape[2]))
    x_min, x_max, y_min, y_max, z_min, z_max = 0, (shape[0]-64), 0, (shape[1]-64), 0,(shape[2] - 64)
    with torch.no_grad():
        for x in tqdm(range(x_min, x_max, 32)):
            for y in range(y_min, y_max, 32):
                for z in range(z_min, z_max, 32):
                    temp = vol[:, x:x+64, y:y+64, z:z+64]
                    temp = np.expand_dims(temp, 0)
                    vol_ = torch.from_numpy(temp).to(device).float()
                    pred = torch.exp(model(vol_).detach()).cpu().numpy()[0]
                    final_prediction[:, x:x+64, y:y+64, z:z+64] = pred
                    
    return final_prediction

def argmax(vol):
    return np.argmax(vol, axis=0)


def get_whole_tumor(data):
    return (data>0)*(data<4)

def get_tumor_core(data):
    return np.logical_or(data==1,data==3)

def get_enhancing_tumor(data):
    return data==4

def get_dice_score(prediction, ground_truth):
    masks=(get_whole_tumor, get_tumor_core, get_enhancing_tumor)
    p    =np.uint8(prediction)
    gt   =np.uint8(ground_truth)
    # print(np.unique(p),np.unique(gt))
    wt,tc,et=[2*np.sum(func(p)*func(gt)) / (np.sum(func(p)) + np.sum(func(gt))+1e-6) for func in masks]
    return wt, tc, et


# In[8]:


net = net.to(device)
LGG_names = ['Brats18_TCIA13_630_1','Brats18_TCIA10_152_1', 'Brats18_TCIA09_620_1', 'Brats18_2013_15_1']
HGG_names = ['Brats18_TCIA03_474_1', 'Brats18_TCIA08_167_1', 'Brats18_2013_10_1', 'Brats18_TCIA02_322_1']
names     =  os.listdir('/home/uavws/pranjal/MICCAI_BraTS_2018_Data_Training/test/LGG')

for name in names:
    root_path = '/home/uavws/pranjal/MICCAI_BraTS_2018_Data_Training/test/LGG/'+name+'/'
    path = root_path + name +'_'

    mask  =  np.uint8(nib.load(root_path+'get_2D_tiramisu.nii.gz').get_data())
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
    final_pred     = inner_class_classification(vol, net) 
    final_pred     = argmax(final_pred)
    # final_pred     = crf(vol, -1*final_pred)
    # final_pred     = perform_postprocessing(final_pred, thresh = 0.5)

    final_pred[final_pred == 4] = 0
    final_pred[final_pred == 3] = 4

    wt, tc, et     = get_dice_score(final_pred, seg)
    print ('wt_dice_score = '+str(wt)+'; tc_dice_score='+str(tc)+'; et_dice_score='+str(et))

    vol = nib.Nifti1Image(final_pred, affine)
    vol.set_data_dtype(np.uint8)
    nib.save(vol, root_path +'3D_Tiramisu_103.nii.gz')
