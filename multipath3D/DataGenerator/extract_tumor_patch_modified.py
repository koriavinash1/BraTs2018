import os
import nibabel as nib
import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import random
import SimpleITK as sitk

# Root Directory
ROOT_DIR = os.path.abspath("../../")

## path is the location where you want to save the segmentation data
def save_img(save_path, matrix_3d, affine):
	directory = os.path.dirname(save_path)

	if not os.path.exists(directory):
		os.makedirs(directory)
	
	img = nib.Nifti1Image(matrix_3d,affine)
	img.set_data_dtype(np.float32) 
	nib.save(img, save_path)

## path is the location of segmentation data to make bounding box
def bounding_box(path):

	# print(path)
 
	seg_file = nib.load(path)

	# affine = seg_file.get_affine()
	seg_data = seg_file.get_data()

	tumor = np.where(seg_data>np.min(seg_data))
	
	x_min = np.min(tumor[0])
	y_min = np.min(tumor[1])
	z_min = np.min(tumor[2])

	x_max = np.max(tumor[0])
	y_max = np.max(tumor[1])
	z_max = np.max(tumor[2])

	# box = np.zeros(np.shape(data))
	# box[x_min:x_max, y_min:y_max, z_min:z_max] = 1
	# plt.imshow(box[:,:,78])
	# plt.show()
	return x_min, x_max, y_min, y_max, z_min, z_max

def normalize2(img):
    mean=np.mean(img[img!=0])
    std=np.std(img[img!=0])
    img_norm=(img-mean)/std
    return img_norm

def normalize(img,mask):
	mean=np.mean(img[mask!=0])
	std=np.std(img[mask!=0])
	return (img-mean)/std

def resize_sitk_3D(image_array, outputSize=None, interpolator=sitk.sitkLinear):
    """
    Resample 3D images Image:
    For Labels use nearest neighbour
    For image use 
    sitkNearestNeighbor = 1,
    sitkLinear = 2,
    sitkBSpline = 3,
    sitkGaussian = 4,
    sitkLabelGaussian = 5, 
    """
    image = sitk.GetImageFromArray(image_array) 
    inputSize = image.GetSize()
    inputSpacing = image.GetSpacing()
    outputSpacing = [1.0, 1.0, 1.0]
    if outputSize:
        outputSpacing[0] = inputSpacing[0] * (inputSize[0] /outputSize[0]);
        outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1]);
        outputSpacing[2] = inputSpacing[2] * (inputSize[2] / outputSize[2]);
    else:
        # If No outputSize is specified then resample to 1mm spacing
        outputSize = [0.0, 0.0, 0.0]
        outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
        outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
        outputSize[2] = int(inputSize[2] * inputSpacing[2] / outputSpacing[2] + .5)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(outputSize)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    image = resampler.Execute(image)
    resampled_arr = sitk.GetArrayFromImage(image)
    return resampled_arr

## path of normalized data, e.g. MICCAI_BraTS_2018_Data_Training/HGG_normalized
def extract_tumor_patches(path):

    _, patients,_ = next(os.walk(path))
    

    ## Looping over all the patient data
    for patient in tqdm(patients):
        total_patches = 0
        # if os.path.exists(path + '_patches/' + patient + '/patch_NON_lesion_0' + '/'):
        #     continue
        print('\npatient id: ',patient)
        ## Loading the nii files
        seg_nifti = nib.load(path + '/' + patient + '/' + patient + '_seg.nii.gz')
        flair_nifti = nib.load(path + '/' + patient + '/' + patient + '_flair.nii.gz')
        t1_nifti = nib.load(path + '/' + patient + '/' + patient + '_t1.nii.gz')
        t1ce_nifti = nib.load(path + '/' + patient + '/' + patient + '_t1ce.nii.gz')
        t2_nifti = nib.load(path + '/' + patient + '/' + patient + '_t2.nii.gz')
        mask = nib.load(path + '/' + patient + '/'+ 'mask.nii.gz')

        ## Getting data from the loaded nifti files (optional: get affine used later for saving in nii format)
        affine = seg_nifti.get_affine()
        seg_data = seg_nifti.get_data()
        flair_data = flair_nifti.get_data()
        t1_data = t1_nifti.get_data()
        t1ce_data = t1ce_nifti.get_data()
        t2_data = t2_nifti.get_data()
        mask   = np.uint8(mask.get_data())
        
        flair_data=normalize(flair_data,mask)
        t1_data=normalize(t1_data,mask)
        t1ce_data=normalize(t1ce_data,mask)
        t2_data=normalize(t2_data,mask)
        
        contrasts_dict = {'flair': flair_data, 
                  't1': t1_data, 
                  't1ce': t1ce_data, 
                  't2': t2_data}

        seg_data[(mask != 0) * (seg_data <= 0)] = 5 #Brain tissue
        ## Set stride for patch incerment, cntr_patch is used to count number of patches for single patient
        stride = 9 ### used to be 6
        cntr_patch = 0
        allowed_classes = [1, 2, 4, 5] # (5 for brain tissue)

        arr = []
        for _class_ in allowed_classes:
            x, y, z      = np.where(seg_data == _class_)
            arr.append([len(x)])

        min_len = np.sort(np.array(arr))[::-1][-1]
        if min_len == 0:
            min_len = np.sort(np.array(arr))[::-1][-2]
        
        for _class_ in tqdm(allowed_classes):
            try:
                x_range, y_range, z_range      = np.where(seg_data == _class_)
                idx = np.random.randint(0, len(x_range), min_len//stride**3)
                print (len(idx))
                cntr = 0
                for (x,y,z) in zip(x_range[idx], y_range[idx], z_range[idx]):
                    seg_patch = seg_data[max(0,x-4):x+5, max(0,y-4):y+5, max(0,z-4):z+5]
                    reduced_seg_patch=np.zeros((9,9,9))
                    if seg_patch.shape!=(9,9,9):
                        x_offset=int((9-seg_patch.shape[0])/2)
                        y_offset=int((9-seg_patch.shape[1])/2)
                        z_offset=int((9-seg_patch.shape[2])/2)
                        reduced_seg_patch[x_offset: x_offset+ seg_patch.shape[0], y_offset: y_offset+ seg_patch.shape[1],
                            z_offset: z_offset+ seg_patch.shape[2]]= seg_patch
                    else:
                        reduced_seg_patch= seg_patch

                    # Checking if the reduced patch contains any tumor

                    path_patch = path +'_patches/' + patient + '/patch_lesion_class' + str(_class_)+'_cntr' + str(cntr) + '/'
                    if os.path.exists(path_patch): continue

                    ## SAVING REDUCED_SEG_PATCH in nii
                    save_img(path_patch + 'seg_9cube' + '.nii.gz', reduced_seg_patch, affine)

                    for contrast in contrasts_dict.keys():

                        ### 25*25*25 High res patch
                        highres = contrasts_dict[contrast][max(0,x-12):x+13, max(0,y-12):y+13, max(0,z-12):z+13]
                        highres_patch=np.zeros((25,25,25))+ np.min(contrasts_dict[contrast])
                        ## Save in nii format
                        if highres.shape !=(25,25,25):
                            x_offset=int((25- highres.shape[0])/2)
                            y_offset=int((25- highres.shape[1])/2)
                            z_offset=int((25- highres.shape[2])/2)
                            highres_patch[x_offset: x_offset+ highres.shape[0], y_offset: y_offset+ highres.shape[1],
                            z_offset: z_offset+ highres.shape[2]]= highres
                        else:
                            highres_patch=highres
                        name_scheme = contrast + '_high_res_25cube' +  '.nii.gz'
                        save_path = os.path.join(path_patch, name_scheme)
                        save_img(save_path, highres_patch, affine)  ## save_img is custom made function

                        #### 51*51*51 reduced to 19*19*19 Low res patch
                        ## adding min value for padding purpose when the patch goes out of image size 
                        lowres_patch = np.zeros((51,51,51)) + np.min(contrasts_dict[contrast])
                        low_res=contrasts_dict[contrast][max(0,x-25):x+26, max(0, y-25):y+26, max(0,z-25):z+26]
                        
                        if low_res.shape !=(51,51,51):
                            x_offset=int((51- low_res.shape[0])/2)
                            y_offset=int((51- low_res.shape[1])/2)
                            z_offset=int((51- low_res.shape[2])/2)
                            lowres_patch[x_offset: x_offset+ low_res.shape[0], y_offset: y_offset+ low_res.shape[1],
                            z_offset: z_offset+ low_res.shape[2]]= low_res
                        else:
                            lowres_patch= low_res
                        
                        downsample_lowres_patch = resize_sitk_3D(lowres_patch, (19,19,19)) 
                        ## Save in nii format
                        save_path = path_patch + contrast + '_low_res_19cube' +  '.nii.gz'
                        save_img(save_path, downsample_lowres_patch, affine)
                        print ("vol saved...")
                        if highres_patch.shape !=(25,25,25) or lowres_patch.shape!=(51,51,51) or reduced_seg_patch.shape!=(9,9,9):
                            print("Alert in tumor! Patient id:{}    Shape of high res={}  Shape of low res={}   Shape of segment".format(patient,
                                highres_patch.shape, lowres_patch.shape, reduced_seg_patch.shape))                       
                    
                    cntr +=1 
                total_patches += cntr
            except:
                continue

        print('patient id: ',patient, 'number of patches: [', total_patches, ']')
    print("..............Completed extracting lesion patch..............")

def main():
    path_training_data = ROOT_DIR + '/MICCAI_BraTS_2018_Data_Training/'
    print (path_training_data)
    extract_tumor_patches(path_training_data + 'LGG')
    extract_tumor_patches(path_training_data + 'HGG')


if __name__ == '__main__':
    main()
