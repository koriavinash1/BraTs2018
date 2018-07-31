import pandas as pd
import numpy as np
import os, random
from tqdm import tqdm
from sklearn.utils import shuffle

train_split = 0.7 # percentage
valid_split = 0.15 # validation split
testing_split = 0.15
root_path   = '/home/uavws/pranjal/MICCAI_BraTS_2018_Data_Training'
# remove pathces conditions if considering raw data

"""
paths = []
training, testing, validation = [], [], []
grade_paths = [_str for _str in next(os.walk(root_path))[1] if _str.__contains__('64_cubes')]


for grade_path in grade_paths:
	print ('Grade',grade_path)
	count    = 0
	path     = os.path.join(root_path, grade_path)
	patients = next(os.walk(path))[1]
	patients = shuffle(patients, random_state=0)
	total    = len(patients)

	for p_id in tqdm(patients):
		p_id_path    = os.path.join(path, p_id)
		paths.append(p_id_path)
		tr, vl, te = False, False, False
		if count < train_split*total:
			tr = True
		elif count < (train_split + valid_split) * total and count >= train_split*total:
			vl = True
		elif count > (train_split + valid_split) * total:
			te = True
		count +=1

		training.append(tr)
		validation.append(vl)
		testing.append(te)

## generate csv
split_df = pd.DataFrame()
split_df['Paths']      = paths
split_df['Training']   = training
split_df['Validation'] = validation
split_df['Testing'] 	 = testing
split_df.to_csv('./train_valid_test_split.csv')
"""


split_df = pd.read_csv('/media/brats/MyPassport/Avinash/Kaminstas_2018/Modified_Kamnistas_Model_2018/DataGenerator/train_valid_test_split.csv')
## all patch path
paths, training, testing, validation = [], [], [], []
root_path = '/media/brats/MyPassport/Avinash/Kaminstas_2018/MICCAI_BraTS_2018_Data_Training'

train_paths, valid_paths, test_paths =[],[], [] 
for path in tqdm(split_df['Paths'].as_matrix()):
	row     = split_df[split_df['Paths'] == path]
	patient_ = path.split('/').pop()
	type_    = path.split('/')[-2]
	if type_.__contains__('LGG'):
		path = os.path.join(root_path, 'LGG_patches', patient_)
	else:
		path = os.path.join(root_path, 'HGG_patches', patient_)
	print (path)
	try:
		patches = next(os.walk(path))[1]
		for patch in patches:
			patch_path = os.path.join(path, patch)
			paths.append(patch_path)
			training.append(row['Training'].values[0])
			validation.append(row['Validation'].values[0])
			testing.append(row['Testing'].values[0])

			if (row['Training'].values[0]) == True:
				train_paths.append(patch_path)
			if (row['Validation'].values[0]) == True:
				valid_paths.append(patch_path)		
			if (row['Testing'].values[0]) == True:
				test_paths.append(patch_path)
	except:
		continue

patch_df = pd.DataFrame()
patch_df['Paths']      = paths
patch_df['Training']   = training
patch_df['Validation'] = validation
patch_df['Testing'] 	 = testing
patch_df.to_csv('./patch_info.csv')

patch_df = pd.DataFrame()
patch_df['Paths']      = train_paths
patch_df.to_csv('./1oversampled_training_patch_info.csv')

patch_df = pd.DataFrame()
patch_df['Paths']      = valid_paths
patch_df.to_csv('./validation_patch_info.csv')

patch_df = pd.DataFrame()
patch_df['Paths']      = test_paths
patch_df.to_csv('./testing_patch_info.csv')
