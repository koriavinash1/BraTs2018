import torch.utils.data as data
import nibabel as nib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch


# def make_dataset(csv_path):

# #"""
# #Takes in a csv path, gives a list comprising of filenames
# #"""
#     paths = []
#     data_reader= pd.read_csv(csv_path)
#     data_reader= data_reader['Paths']
#     data_reader= np.array(data_reader)
#     for  fnames in range(len(data_reader)):
#          paths.append(data_reader[fnames])

#     return paths


def nii_loader(path):
	#"""
	#Now given a path, we have to load
	#1) High res version of flair,t2,t1 and t1c
	#2) Low  res version of flair,t2,t1 and t1c
	#3) load the segmentation mask too.


	#Accepts
	#1) path of data

	#Returns
	#1) a 4D high resolution data
	#2) a 4d low resolution data
	#3) Segmentation
	#"""

	flair_high_res = nib.load(path+'/'+'flair_high_res_25cube.nii.gz').get_data()
	flair_low_res  = nib.load(path+'/'+'flair_low_res_19cube.nii.gz').get_data()

	t2_high_res = nib.load(path+'/'+'t2_high_res_25cube.nii.gz').get_data()
	t2_low_res  = nib.load(path+'/'+'t2_low_res_19cube.nii.gz').get_data()

	t1_high_res = nib.load(path+'/'+'t1_high_res_25cube.nii.gz').get_data()
	t1_low_res  = nib.load(path+'/'+'t1_low_res_19cube.nii.gz').get_data()

	t1ce_high_res = nib.load(path+'/'+'t1ce_high_res_25cube.nii.gz').get_data()
	t1ce_low_res  = nib.load(path+'/'+'t1ce_low_res_19cube.nii.gz').get_data()



	sege_mask = np.uint8(nib.load(path+'/'+'seg_9cube.nii.gz').get_data())  ## converting the labels betwen 0-N done in below!!! Beware. 



	high_resolution_data = np.zeros((4,25,25,25))
	low_resolution_data  = np.zeros((4,19,19,19))

	try:
		shape = flair_high_res.shape
		high_resolution_data[0,:,:,:]=flair_high_res
		high_resolution_data[1,:,:,:]=t2_high_res
		high_resolution_data[2,:,:,:]=t1_high_res
		high_resolution_data[3,:,:,:]=t1ce_high_res

		shape = flair_low_res.shape
		low_resolution_data[0,:,:,:]=flair_low_res
		low_resolution_data[1,:,:,:]=t2_low_res
		low_resolution_data[2,:,:,:]=t1_low_res
		low_resolution_data[3,:,:,:]=t1ce_low_res
		# print ("try successful")
	except:
		print(path)
	return high_resolution_data,low_resolution_data,sege_mask

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

    def __init__(self, imgs, transform=None, target_transform=None,
                 loader=nii_loader):


        # imgs = make_dataset(root) ### root= entire csv path
        self.imgs = imgs#[:16]
        np.random.shuffle(self.imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path= self.imgs[index]
        high_res_input, low_res_input, segmentation = self.loader(path)
        segmentation[segmentation==4]=3   ### Attempt to make dataset between 0-N. enhancing set from 4 to 3
        segmentation[segmentation==5]=4   ### Attempt to make dataset between 0-N. enhancing set from 5 to 4
        high_res_input= torch.from_numpy(high_res_input).float() ## torchifying the data.
        low_res_input = torch.from_numpy(low_res_input).float() ## torchifying the data.

        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return high_res_input,low_res_input, segmentation, path

    def __len__(self):
        return len(self.imgs)


# h_data, l_data, seg = nii_loader(f[100])

# plt.imshow(h_data[0,:,:,10],cmap='gray')
# plt.figure()
# plt.imshow(l_data[0,:,:,10],cmap='gray')
# plt.show()

def getDataPaths(path):
    data = pd.read_csv(path)
    imgpaths = data['Paths'].as_matrix()
    np.random.shuffle(imgpaths)
    return imgpaths


if __name__== '__main__':

    csv='./validation_patch_info.csv'
    a= getDataPaths(csv)

    dl=ImageFolder(a)
    for i, (x,y,z,_) in enumerate (dl):
        # print ('min',torch.min(z),'max',torch.max(z), _)
        print('size',z.long().size())
