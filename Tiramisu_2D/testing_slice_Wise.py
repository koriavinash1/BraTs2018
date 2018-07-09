
# coding: utf-8

# In[1]:


import os
import nibabel as nib
import numpy as np
import torch
from torchvision import transforms
from model import FCDenseNet103, FCDenseNet57, FCDenseNet67
from tqdm import tqdm

# ### Loading the model

# In[4]:



model = FCDenseNet57(n_classes = 4) ## intialize the graph
checkpoint= '/media/brats/MyPassport/Avinash/models/model-m-23062018-103819-tramisu_2D_FC57_loss = 0.19099325378230858_acc = 0.9974654925456413_best_loss.pth.tar' ## path to the model to be tested
saved_parms=torch.load(checkpoint) ## load the params
model.load_state_dict(saved_parms['state_dict']) ## fill the model with trained params
model.eval()



# ## Define the Transforms required just once

# In[ ]:


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformSequence=transforms.Compose(transformList)


# ## Define mode of running GPU/CPU

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# ## Define normalization function. Slice-wise Normalization

# In[ ]:


def scale_every_slice_between_0_to_255(a):
    normalized_a=  255*((a-np.min(a))/(np.max(a)-np.min(a)))
    return normalized_a
    


# ### Define the path containing the test data.
# 
#             

# In[ ]:


with torch.no_grad():   ## required to minimize GPU usage during inference.
    volume_path ='/media/brats/MyPassport/Avinash/Kaminstas_2018/MICCAI_BraTS_2018_Data_Training/test'
    patient_id = os.listdir(volume_path)
    for p in patient_id:
        print ('patient_id',p)
        sequences = os.listdir(volume_path+'/'+p)
        for s in sequences:
            if 'flair' in s:
                flair_volume= nib.load(volume_path+'/'+p+'/'+s)
            if 't2' in s:
                t2_volume= nib.load(volume_path+'/'+p+'/'+s)
            if 't1ce' in s:
                t1ce_volume= nib.load(volume_path+'/'+p+'/'+s)
        affine = flair_volume.get_affine()
        flair_volume= flair_volume.get_data()
        t2_volume   = t2_volume.get_data()
        t1ce_volume = t1ce_volume.get_data()
        generated_output = np.empty((4, flair_volume.shape[0],flair_volume.shape[1],flair_volume.shape[2]))
        for slices in tqdm(range(flair_volume.shape[2])):
            flair_slice= scale_every_slice_between_0_to_255(np.transpose(flair_volume[:,:,slices]))
            t2_slice= scale_every_slice_between_0_to_255(np.transpose(t2_volume[:,:,slices]))
            t1ce_slice= scale_every_slice_between_0_to_255(np.transpose(t1ce_volume[:,:,slices]))
            array=np.zeros((flair_slice.shape[0],flair_slice.shape[1],3))
            array[:,:,0]=flair_slice
            array[:,:,1]=t2_slice
            array[:,:,2]=t1ce_slice
            array = np.uint8(array) ## making it int so that To_Tensor works
            transformed_array = transformSequence(array)
            transformed_array=transformed_array.unsqueeze(0) ## neccessary if batch size == 1
            transformed_array= transformed_array.to(device)
            outs = model(transformed_array)
            _,preds= torch.max(outs,1)
            # del outs
            del _
            preds = preds.data.cpu().numpy()  ## variable converted-->cpu---> numpy
            preds[preds==3]=4       ## converting to 0,1,2,4 
            generated_output[:,:,:,slices] = torch.exp(outs[0]).data.cpu().numpy() ## fill the slice
        
        # generated_output= np.uint8(generated_output)
        generated_output= np.swapaxes(generated_output,2, 1)
        # 3D connected comp...
        imgs = nib.Nifti1Image(generated_output, affine)
        imgs.set_data_dtype(np.float32)
        nib.save(imgs,volume_path+'/'+p+'/get_2D_tiramisu_softmax.nii.gz')
        
        

