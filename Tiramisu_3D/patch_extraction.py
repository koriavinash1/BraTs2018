import os
import nibabel as nib
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
# Root Directory
ROOT_DIR = os.path.abspath("../")

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

	return x_min, x_max, y_min, y_max, z_min, z_max

def normalize2(img):
    mean=np.mean(img[img!=0])
    std=np.std(img[img!=0])
    img_norm=(img-mean)/std
    return img_norm

def normalize(img, mask):
	mean  =np.mean(img[mask !=0 ])
	std   =np.std(img[mask !=0 ])
	return (img-mean)/std

## path of normalized data, e.g. MICCAI_BraTS_2018_Data_Training/HGG_normalized
def extract_tumor_patches(path):
	print (path)
	_, patients,_ = next(os.walk(path))

	## Looping over all the patient data
	for patient in tqdm(patients):
		# if os.path.exists(path + '_patches/' + patient + '/patch_NON_lesion_0' + '/'):
		#     continue
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

		flair_data =normalize(flair_data, mask)
		t1_data    =normalize(t1_data, mask)
		t1ce_data  =normalize(t1ce_data, mask)
		t2_data    =normalize(t2_data, mask)

		contrasts_dict = {'flair': flair_data,
							't1': t1_data,
							't1ce': t1ce_data,
							't2': t2_data}

		seg_data[(mask != 0) * (seg_data <= 0)] = 5

		## Set stride for patch incerment, cntr_patch is used to count number of patches for single patient
		stride = 6  ## used to be 6
		cntr_patch = 0

		allowed_classes = [1, 2, 4, 5] # (5 for brain tissue)

		arr = []
		for _class_ in allowed_classes:
			x, y, z      = np.where(seg_data == _class_)
			arr.append([len(x)])

		min_len = np.sort(np.array(arr))[::-1][-1]
		if min_len == 0:
			min_len = np.sort(np.array(arr))[::-1][-2]
			# allowed_classes = np.delete(allowed_classes, np.where(arr == min_len))

		total_cntr = 0
		for _class_ in tqdm(allowed_classes):
			x_range, y_range, z_range      = np.where(seg_data == _class_)
			try:
				idx = np.random.randint(0, len(x_range), size = min_len//stride**3) ### used to be square
				# print (len(idx), idx)
				cntr = 0
				for (x,y,z) in zip(x_range[idx], y_range[idx], z_range[idx]):
					seg_mask = np.zeros((64,64,64))
					_mask   = np.uint8(seg_data[max(0, x-31):min(240, x+33), max(0,y-31):min(240, y+33), max(0, z-31):min(240, z+33)])
					saver_path='/home/uavws/pranjal/MICCAI_BraTS_2018_Data_Training/'
					path_patch = saver_path + 'HGG_64_cubes_high_sampling/' + patient + '/patch_lesion_class' + str(_class_) + '_cntr' + str(cntr) + '/'
					## SAVING REDUCED_SEG_PATCH in nii
					os.makedirs(path_patch,exist_ok=True)
					if _mask.shape != (64,64,64):
						x_offset = int((64 - _mask.shape[0])/2)
						y_offset = int((64 - _mask.shape[1])/2)
						z_offset = int((64 - _mask.shape[2])/2)
						seg_mask[x_offset: x_offset+_mask.shape[0], y_offset: y_offset+ _mask.shape[1], z_offset: z_offset+ _mask.shape[2]] = _mask  
					else:
						seg_mask=_mask

					save_img(path_patch + 'seg_64cube' + '.nii.gz', seg_mask, affine)
					for contrast in contrasts_dict.keys():
						vol_mask = np.zeros((64,64,64)) + np.min(contrasts_dict[contrast])# min background
						vol = np.float32(contrasts_dict[contrast][max(0, x-31):min(240, x+33), max(0,y-31):min(240, y+33), max(0, z-31):min(240, z+33)])
						if _mask.shape != (64,64,64):
							x_offset = int((64 - _mask.shape[0])/2)
							y_offset = int((64 - _mask.shape[1])/2)
							z_offset = int((64 - _mask.shape[2])/2)
							vol_mask[x_offset: x_offset+_mask.shape[0], y_offset: y_offset+ _mask.shape[1], z_offset: z_offset+ _mask.shape[2]] = vol
						else:
							vol_mask=vol
	
						name_scheme = contrast + '_64cube' +  '.nii.gz'
						save_path   = os.path.join(path_patch, name_scheme)
						if seg_mask.shape != (64,64,64) or vol_mask.shape != (64,64,64): print ("alert", patient, 'shape: ', seg_mask.shape)
						save_img(save_path, vol_mask, affine)
						
					cntr += 1
				total_cntr += cntr
			except:
				continue
		print('patient id: ',patient, 'number of patches: [', total_cntr,"]")

	print("..............Completed extracting patch..............")

def main():
	path_training_data = ROOT_DIR + '/MICCAI_BraTS_2018_Data_Training/'
	extract_tumor_patches(path_training_data + 'HGG')
	# extract_tumor_patches(path_training_data + 'HGG')


if __name__ == '__main__':
	main()
