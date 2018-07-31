import pandas as pd 
import numpy as np 
import os 
from tqdm import tqdm

csv_path = '/home/uavws/pranjal/dummy.csv'
CSV = pd.read_csv(csv_path)
paths =np.array(CSV['Paths'])

Training_flag = np.array(CSV['Training'])
Validation_flag = np.array(CSV['Validation'])
Testing_flag = np.array(CSV['Testing'])

training_list=[]
validation_list=[]
testing_list=[]
for i in tqdm(range(len(Training_flag))):
	if Training_flag[i] == True:
		training_list.append(paths[i])
	if Validation_flag[i] == True:
		validation_list.append(paths[i])
	if Testing_flag[i] == True:
		testing_list.append(paths[i])

patch_df = pd.DataFrame()
patch_df['Paths']      = paths
patch_df['Training']   = training
# patch_df['Validation'] = validation
# patch_df['Testing'] 	 = testing
patch_df.to_csv('./patch_info.csv')
