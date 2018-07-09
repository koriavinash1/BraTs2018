import torch.utils.data as data
import nibabel as nib
import pandas as pd
import h5py
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.feature import canny
import matplotlib.pyplot as plt
from cv2 import bilateralFilter
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_tv_bregman
from skimage.transform import resize
import SimpleITK as sitk
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

def hdf5_loader(path):
    h5 = h5py.File(path)
    # data = np.float32(h5['Sequence'][:]).transpose(2,0,1)# seq has flair, t2, t1ce
    data = np.uint8(h5['Sequence'][:])# seq has flair, t2, t1ce,t1
    # data = np.swapaxes(data, 0, 2)/255.0
    seg = np.int16(h5['label'][:])
    seg[seg == 4] = 3 # 0, 1,2,3
    # print (data.dtype, seg.dtype)
    return data, seg


def multilabel_binarize(image_nD, nlabel):
    """
    Binarize multilabel images and return stack of binary images
    Returns: Tensor of shape: Bin_Channels* Image_shape(3D tensor)
    TODO: Labels are assumed to discreet in steps from -> 0,1,2,...,nlabel-1
    """
    labels = range(nlabel)
    out_shape = (len(labels),) + image_nD.shape
    bin_img_stack = np.ones(out_shape, dtype='uint8')
    for label in labels:
        bin_img_stack[label] = np.where(image_nD == label, bin_img_stack[label], 0)
    return bin_img_stack


selem = morph.disk(1)
def getEdgeEnhancedWeightMap(label, label_ids =[0,1,2,3], scale=1, edgescale=1, assign_equal_wt=False):
    label = multilabel_binarize(label, len(label_ids))# convert to onehot vector
    shape = (0,)+label.shape[1:]
    weight_map = np.empty(shape, dtype='uint8')
    if assign_equal_wt:
        return np.ones_like(label)
    for i in range(label.shape[0]): 
        #Estimate weight maps:
        weights = np.ones(len(label_ids))
        slice_map = np.ones(label[i,:,:].shape)
        for _id in label_ids:
            class_frequency = np.sum(label[i,:,:] == label_ids[_id])
            if class_frequency:
                weights[label_ids.index(_id)] = scale*label[i,:,:].size/class_frequency
                slice_map[np.where(label[i,:,:]==label_ids.index(_id))] = weights[label_ids.index(_id)]
                edge = np.float32(morph.binary_dilation(
                    canny(np.float32(label[i,:,:]==label_ids.index(_id)),sigma=1), selem=selem))
                edge_frequency = np.sum(np.sum(edge==1.0))
                if edge_frequency:    
                    slice_map[np.where(edge==1.0)] += edgescale*label[i,:,:].size/edge_frequency
            # print (weights)
            # utils.imshow(edge, cmap='gray')
        # utils.imshow(weight_map, cmap='gray')
        weight_map = np.append(weight_map, np.expand_dims(slice_map, axis=0), axis=0)
    # plt.imshow(np.mean(weight_map, 0))
    # plt.show()
    return np.float32(np.mean(weight_map, 0))

class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root_path, transform=None,
                 loader=hdf5_loader):


        imgs = next(os.walk(root_path))[2]
        self.imgs = imgs #[:100]
        # np.random.shuffle(self.imgs)
        self.transform = transform
        self.loader = loader
        self.root_path = root_path

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path= self.imgs[index]
        input_, segmentation = self.loader(os.path.join(self.root_path, path))
        # print (np.max(input_), np.min(input_), input_.shape)
        weight_map = getEdgeEnhancedWeightMap(segmentation)
        # input_= torch.from_numpy(input_).float() ## torchifying the data.


        if self.transform is not None:
            input_ = self.transform(input_)
        # input_ = torch.FloatTensor(input_)
        return input_, segmentation, weight_map, path

    def __len__(self):
        return len(self.imgs)


if __name__== '__main__':
    a = '/media/brats/MyPassport/Avinash/Kaminstas_2018/MICCAI_BraTS_2018_Data_Training/hdf5_data/train_set'
    normalize = transforms.Normalize([0.4465654,  0.39885548, 0.382155,   0.6104794],[0.1427262,  0.14297858, 0.09494335, 0.13883562])
    transformList = []
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transformSequence=transforms.Compose(transformList)
    dl=ImageFolder(a,transformSequence)
    
    dataLoaderTrain = DataLoader(dataset=dl, batch_size=5, shuffle=True,  num_workers=8, pin_memory=False)
    ino,seg,weight_map,_= next(iter(dataLoaderTrain))
    # plt.imshow(weight)

    # for  high_res, seg, weight_map,  _ in tqdm(dataLoaderTrain):
    #     pass



    # for i, (x,z,w,_) in enumerate (dl):
    
    #     print ('min',np.min(x),'max',np.max(x), _)
    #     cc = torchvision.transforms.ToTensor()(x)
    #     print ('max', torch.max(cc), 'min', torch.min(cc))
        # print('size',z.long().size())
