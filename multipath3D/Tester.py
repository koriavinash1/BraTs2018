
# coding: utf-8

# In[1]:


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
from scipy.ndimage import uniform_filter, maximum_filter
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




#========================================================================================
from model import BrainNet_3D, BrainNet_3D_Inception
net = BrainNet_3D_Inception()

ckpt = torch.load('/media/brats/MyPassport/Avinash/Kaminstas_2018/Modified_Kamnistas_Inception_IN_Layer_Model_2018/models/model-m-05072018-004948-BrainNet_3D_Inception_layer_in_LAYERS_loss = 0.5190369295413275_acc = 0.79661275616758_best_acc.pth.tar')
net.load_state_dict(ckpt['state_dict'])

nclasses = 5
net.eval()
net = net.to(device)

#========================================================================================

from tiramisu import FCDenseNet103, FCDenseNet57, FCDenseNet67
mask_net = FCDenseNet103(n_classes = 3) ## intialize the graph
checkpoint= '/media/brats/0d4a2225-d6b1-4b80-94fd-7c8ae0b1fa102/varghese/brats_2018/zscore_src/models_zscore_lesion_only_3_classes_103/CE_best_model_loss_based.pth.tar' 
saved_parms=torch.load(checkpoint) ## load the params
mask_net.load_state_dict(saved_parms['state_dict']) ## fill the model with trained params
mask_net = mask_net.to(device)
mask_net.eval()


# In[6]:

def bbox(vol):
    # print (np.unique(vol))
    tumor = np.where(vol>0)
    # print (tumor)
    x_min = np.min(tumor[0])
    y_min = np.min(tumor[1])
    z_min = np.min(tumor[2])

    x_max = np.max(tumor[0])
    y_max = np.max(tumor[1])
    z_max = np.max(tumor[2])
    
    return x_min, x_max, y_min, y_max, z_min, z_max        


# def perform_postprocessing(vol, thresh = 0.750, verbose=False):
#     label_vol, label_masks = label(vol)
#     # print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#     # print (np.unique(label_vol))
#     volumes = []
#     for i in range(label_masks):
#         x, y, z = np.where(label_vol == i)
#         v = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y)) * (np.max(z) - np.min(z))
#         volumes.append(v)

#     if verbose: print ("Max. Volume: {}".format(np.max(volumes)) + " Min. Volumes: {}".format(np.min(volumes)))

#     for i in range(len(volumes)):
#         if volumes[i] < np.max(volumes) * thresh:
#             label_vol[label_vol == i] = 0

#     label_vol[label_vol > 0] = 1
#     return label_vol * vol


def perform_postprocessing(voxels, threshold=12000):
    c,n = label(voxels)
    nums = np.array([np.sum(c==i) for i in range(1, n+1)])
    selected_components = nums>threshold
    selected_components[np.argmax(nums)] = True
    mask = np.zeros_like(voxels)
    print(selected_components.tolist())
    for i,select in enumerate(selected_components):
        if select:
            mask[c==(i+1)]=1
    return mask*voxels

def normalize(img,mask):
    mean=np.mean(img[mask!=0])
    std=np.std(img[mask!=0])
    return (img-mean)/std

def adjust_classes_air_brain_tumour(volume):
    ""
    volume[volume == 1] = 0
    volume[volume == 2] = 1
    return volume

def convert_image(image):
    ""
    x= np.float32(image)
    x=np.swapaxes(x,0,2)
    x= np.swapaxes(x,1,2)
    return x


def apply_argmax_to_logits(logits):
    "logits dimensions: nclasses, height, width, depth"
    logits = np.argmax(logits, axis=0)         
    return np.uint8(logits)


def adjust_classes(volume):
    ""
    volume[volume == 4] = 0
    volume[volume == 3] = 4
    return volume


def save_volume(volume, affine, path):
    volume = nib.Nifti1Image(volume, affine)
    volume.set_data_dtype(np.uint8)
    nib.save(volume, path +'.nii.gz')
    pass


def class_wise_postprocessing(logits):
    "logits dimension: nclasses, width, height, depth"
    return_ = np.zeros_like(logits)
    for class_ in range(logits.shape[0]):
        return_[class_, :, :, :] = perform_postprocessing(logits[class_, :, :, :])

    return return_

def postprocessing_pydensecrf(logits):
    # probs of shape 3d image per class: Nb_classes x Height x Width x Depth
    shape = logits.shape[1:]
    new_image = np.empty(shape)
    d = dcrf.DenseCRF(np.prod(shape), logits.shape[0])
    U = unary_from_softmax(logits)
    d.setUnaryEnergy(U)
    feats = create_pairwise_gaussian(sdims=(1.0, 1.0, 1.0), shape=shape)
    d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5) 
    new_image = np.argmax(Q, axis=0).reshape((shape[0], shape[1],shape[2]))
    return new_image


# def inner_class_classification_with_logits(t1, t1ce, t2, flair, model):
#     shape = t1.shape # to exclude batch_size
#     final_prediction_logits = np.zeros((nclasses, shape[0], shape[1], shape[2]))

#     # x_min, x_max, y_min, y_max, z_min, z_max = bbox(mask)
#     x_min, x_max, y_min, y_max, z_min, z_max = 0, 231, 0, 231, 0, 146
#     contrasts_dict = {'t1':t1, 't1ce': t1ce, 't2': t2, 'flair': flair}

#     for x in tqdm(range(x_min, x_max, 9)):
#         for y in range(y_min, y_max, 9):
#             for z in range(z_min, z_max, 9):
#                 high = np.zeros((1, 4, 25, 25, 25))
#                 low  = np.zeros((1, 4, 19, 19, 19))

#                 for i, contrast in enumerate(contrasts_dict.keys()):

#                     temp = contrasts_dict[contrast][max(0,x-12):x+13, max(0,y-12):y+13, max(0,z-12):z+13]
#                     highres_patch = np.zeros((25, 25, 25))+ np.min(contrasts_dict[contrast])
#                     if temp.shape !=(25,25,25):
#                         x_offset=int((25- temp.shape[0])/2)
#                         y_offset=int((25- temp.shape[1])/2)
#                         z_offset=int((25- temp.shape[2])/2)
#                         highres_patch[x_offset: x_offset + temp.shape[0], y_offset: y_offset + temp.shape[1], z_offset: z_offset+ temp.shape[2]] = temp
#                     else:
#                         highres_patch = temp
                    
#                     high[0, i, :, :, :] = highres_patch

#                     temp  = contrasts_dict[contrast][max(0,x-25):x+26, max(0, y-25):y+26, max(0,z-25):z+26]
#                     lowres_patch = np.zeros((51, 51, 51))+ np.min(contrasts_dict[contrast])
#                     if temp.shape !=(51, 51, 51):
#                         x_offset=int((51 - temp.shape[0])/2)
#                         y_offset=int((51 - temp.shape[1])/2)
#                         z_offset=int((51 - temp.shape[2])/2)
#                         lowres_patch[x_offset: x_offset + temp.shape[0], y_offset: y_offset + temp.shape[1], z_offset: z_offset+ temp.shape[2]] = temp
#                     else:
#                         lowres_patch = temp

#                     low[0, i, :, :, :] = resize(lowres_patch, (19,19,19))
                
#                 high = Variable(torch.from_numpy(high)).to(device).float()
#                 low  = Variable(torch.from_numpy(low)).to(device).float()
#                 pred = model(high, low).detach().cpu().numpy()

#                 final_prediction_logits[:, x:x+9, y:y+9, z:z+9] = pred[0] # 0 is just to remove batchsize axis
    
#     final_prediction_logits = class_wise_postprocessing(final_prediction_logits)
#     return final_prediction_logits


def get_localization(t1_v, t1c_v, t2_v, flair_v):
    generated_output_logits = np.empty((3, flair_v.shape[0],flair_v.shape[1],flair_v.shape[2]))

    for slices in tqdm(range(flair_v.shape[2])):
        flair_slice= np.transpose(flair_v[:,:,slices])
        t2_slice= np.transpose(t2_v[:,:,slices])
        t1ce_slice= np.transpose(t1c_v[:,:,slices])
        t1_slice= np.transpose(t1_v[:,:,slices])  
                      
        array=np.zeros((flair_slice.shape[0],flair_slice.shape[1],4))
        array[:,:,0]=flair_slice
        array[:,:,1]=t2_slice
        array[:,:,2]=t1ce_slice
        array[:,:,3]=t1_slice
            

        transformed_array = torch.from_numpy(convert_image(array)).float()
        transformed_array=transformed_array.unsqueeze(0) ## neccessary if batch size == 1
        transformed_array= transformed_array.to(device)
        logits = mask_net(transformed_array).detach().cpu().numpy()# 3 x 240 x 240  
        
        generated_output_logits[:,:,:, slices] = logits

    # logitsC     = class_wise_postprocessing(generated_output_logits)
    # final_pred  = postprocessing_pydensecrf(logitsC)
    final_pred  = apply_argmax_to_logits(generated_output_logits)
    final_pred  = adjust_classes_air_brain_tumour(np.uint8(final_pred))
    return np.uint8(final_pred)



def inner_class_classification_with_logits(t1, t1ce, t2, flair, mask, model):
    shape = t1.shape # to exclude batch_size
    final_prediction = np.zeros((nclasses, shape[0], shape[1], shape[2]))
    x_min, x_max, y_min, y_max, z_min, z_max = bbox(mask)
    #x_min, x_max, y_min, y_max, z_min, z_max = 25, 210, 25, 210, 25, 120
    print (x_min, x_max, y_min, y_max, z_min, z_max)
    x_min, x_max, y_min, y_max, z_min, z_max = max(25, x_min), min(210, x_max), max(25, y_min), min(210, y_max), max(25, z_min), min(120, z_max)
    for x in tqdm(range(x_min, x_max, 9)):
        for y in range(y_min, y_max, 9):
            for z in range(z_min, z_max, 9):
                high = np.zeros((1, 4, 25, 25, 25))
                low  = np.zeros((1, 4, 19, 19, 19))

                high[0, 0, :, :, :] = flair[x-8:x+17, y-8:y+17, z-8:z+17]
                high[0, 1, :, :, :] = t2[x-8:x+17, y-8:y+17, z-8:z+17]
                high[0, 2, :, :, :] = t1[x-8:x+17, y-8:y+17, z-8:z+17]
                high[0, 3, :, :, :] = t1ce[x-8:x+17, y-8:y+17, z-8:z+17]

                low[0, 0, :, :, :]  = resize(flair[x-21:x+30, y-21:y+30, z-21:z+30], (19,19,19))
                low[0, 1, :, :, :]  = resize(t2[x-21:x+30, y-21:y+30, z-21:z+30], (19,19,19))
                low[0, 2, :, :, :]  = resize(t1[x-21:x+30, y-21:y+30, z-21:z+30], (19,19,19))
                low[0, 3, :, :, :]  = resize(t1ce[x-21:x+30, y-21:y+30, z-21:z+30], (19,19,19))
                
                high = Variable(torch.from_numpy(high)).to(device).float()
                low  = Variable(torch.from_numpy(low)).to(device).float()
                pred = model(high, low).detach().cpu().numpy()

                final_prediction[:, x:x+9, y:y+9, z:z+9] = pred[0]
                
    return final_prediction

def get_whole_tumor(data):
    return (data>0)*(data<4)

def get_tumor_core(data):
    return np.logical_or(data==1,data==3)

def get_enhancing_tumor(data):
    return data==4

def get_dice_score(prediction, ground_truth):
    # print (np.unique(prediction), np.unique(ground_truth))
    masks=(get_whole_tumor, get_tumor_core, get_enhancing_tumor)
    p    =np.uint8(prediction)
    gt   =np.uint8(ground_truth)
    wt,tc,et=[2*np.sum(func(p)*func(gt)) / (np.sum(func(p)) + np.sum(func(gt))+1e-6) for func in masks]
    return wt, tc, et


# names     =  os.listdir('/media/brats/MyPassport/Avinash/Kaminstas_2018/MICCAI_BraTS_2018_Data_Training/test/HGG')

# for name in names:
#     root_path = '/media/brats/MyPassport/Avinash/Kaminstas_2018/MICCAI_BraTS_2018_Data_Training/test/HGG/'+name+'/'
#     path = root_path + name +'_'


#     brain_mask = np.uint8(nib.load(root_path + 'mask.nii.gz').get_data())
#     seg   =  np.uint8(nib.load(path+'seg.nii.gz').get_data())

#     flair =  normalize(nib.load(path+'flair.nii.gz').get_data(), brain_mask)
#     t1    =  normalize(nib.load(path+'t1.nii.gz').get_data(), brain_mask)
#     t1ce  =  normalize(nib.load(path+'t1ce.nii.gz').get_data(), brain_mask)
#     t2    =  normalize(nib.load(path+'t2.nii.gz').get_data(), brain_mask)
#     affine=  nib.load(path+'seg.nii.gz').affine
#     print (path)




names     =  os.listdir('/media/brats/MyPassport/MICCAI_BraTS_2018_Data_Validation')
root_save      = '/media/brats/MyPassport/ValidationPredictions_Kam3D/'
os.makedirs(root_save, exist_ok=True)
for name in tqdm(names):
    root_path = '/media/brats/MyPassport/MICCAI_BraTS_2018_Data_Validation/'+name+'/'
    path = root_path + name + '_'

    # mask  =  np.uint8(nib.load(root_path+'get_2D_tiramisu.nii.gz').get_data())
    brain_mask = np.uint8(nib.load(root_path + 'mask.nii.gz').get_data())
    # seg   =  np.uint8(nib.load(path+'seg.nii.gz').get_data())

    flair =  normalize(nib.load(path+'flair.nii.gz').get_data(), brain_mask)
    t1    =  normalize(nib.load(path+'t1.nii.gz').get_data(), brain_mask)
    t1ce  =  normalize(nib.load(path+'t1ce.nii.gz').get_data(), brain_mask)
    t2    =  normalize(nib.load(path+'t2.nii.gz').get_data(), brain_mask)
    affine=  nib.load(path+'flair.nii.gz').affine
    print (path)
    mask  =  get_localization(t1, t1ce, t2, flair)
    mask  =  np.swapaxes(mask,1, 0)

    final_prediction_logits = inner_class_classification_with_logits(t1, t1ce, t2, flair, mask, net)
    # final_prediction_logitsC = class_wise_postprocessing(final_prediction_logits)


    # final_pred              = apply_argmax_to_logits(final_prediction_logits)
    # final_pred              = adjust_classes(final_pred)
    # wt, tc, et              = get_dice_score(final_pred, seg)
    # print ("No PostProcessing    ", 'wt_dice_score = ' +str(wt)+'; tc_dice_score='+str(tc)+'; et_dice_score='+str(et))
    # save_volume(final_pred, affine, root_path +'3D_Kam_Incep_with_loc_models_No_postprocessing')    

    # final_pred              = apply_argmax_to_logits(final_prediction_logits)
    # final_pred              = perform_postprocessing(final_pred)
    # final_pred              = adjust_classes(final_pred)
    # wt, tc, et              = get_dice_score(final_pred, seg)
    # print ("volume thresh PostProcessing    ", 'wt_dice_score = ' +str(wt)+'; tc_dice_score='+str(tc)+'; et_dice_score='+str(et))
    # save_volume(final_pred, affine, root_path +'3D_Kam_Incep_with_loc_vol_thresh_postprocessing')  


    final_pred              = apply_argmax_to_logits(final_prediction_logits)
    final_pred              = perform_postprocessing(final_pred)
    final_pred              = adjust_classes(final_pred)
    # wt, tc, et              = get_dice_score(final_pred, seg)
    # print ("ClassWisePostProcessing    ", 'wt_dice_score = ' +str(wt)+'; tc_dice_score='+str(tc)+'; et_dice_score='+str(et))
    # save_volume(final_pred, affine, root_path +'3D_Kam_Incep_with_loc_class_wise_vol_thresh_postprocessing')   
    save_volume(final_pred, affine, root_save + name)   


    # final_pred              = postprocessing_pydensecrf(final_prediction_logits)
    # final_pred              = adjust_classes(final_pred)
    # wt, tc, et              = get_dice_score(final_pred, seg)
    # print ("postprocessing_pydensecrf    ", 'wt_dice_score = ' +str(wt)+'; tc_dice_score='+str(tc)+'; et_dice_score='+str(et))
    # save_volume(final_pred, affine, root_path +'3D_Kam_Incep_with_loc_pydensecrf_postprocessing.nii.gz')    


    # final_pred              = postprocessing_pydensecrf(final_prediction_logits)
    # final_pred              = perform_postprocessing(final_pred, thresh = 0.5)
    # final_pred              = adjust_classes(final_pred)
    # wt, tc, et              = get_dice_score(final_pred, seg)
    # print ("postprocessing_pydensecrf + volumebased    ", 'wt_dice_score = ' +str(wt)+'; tc_dice_score='+str(tc)+'; et_dice_score='+str(et))
    # save_volume(final_pred, affine, root_path +'3D_Kam_Incep_with_loc_pydensecrf+volume_thresh_postprocessing') 


    # final_pred              = postprocessing_pydensecrf(final_prediction_logitsC)
    # final_pred              = perform_postprocessing(final_pred, thresh = 0.5)
    # final_pred              = adjust_classes(final_pred)
    # wt, tc, et              = get_dice_score(final_pred, seg)
    # print ("postprocessing_pydensecrf + classwisevolumebased    ", 'wt_dice_score = ' +str(wt)+'; tc_dice_score='+str(tc)+'; et_dice_score='+str(et))
    # save_volume(final_pred, affine, root_path +'3D_Kam_Incep_with_loc_pydensecrf+classwisevolume_thresh_postprocessing')    
