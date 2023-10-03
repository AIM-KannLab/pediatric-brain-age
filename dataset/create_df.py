
#pid,label,dir
import sys
# prepare your own data
import pandas as pd
import nibabel as nib
import numpy as np
import os
from sklearn.model_selection import train_test_split

print( "Reading in csv files ...")
max_age = 35.0*12
min_age = 3.0


centile = 75
print("dataset/csv_data/2D_Dataset_skull_stripped"+str(centile)+"_test2.csv")
df = pd.read_csv("dataset/csv_data/2D_Dataset_skull_stripped"+str(centile)+"_test2.csv", delimiter="," , header=None)
print( "Done reading in csv files ...", df.shape)

def create_dataset(input_df):
    #n_of_baby = 400 # 400 was before - check if unique
    counter = 0
    list_train, list_test_healthy, list_test_cancer, list_test_long, y_train, y_test, y_train_strat = [],[],[],[],[],[],[]
    n_per_age = 100
    
    age_bins = [ [] for _ in range(36)]
    for i in range(input_df.shape[0]):
        input_image_path = input_df[0].iloc[i] 
        split = input_df[3].iloc[i]
        dataset = input_df[4].iloc[i]
        
        if os.path.exists(input_image_path):
            age = input_df[1].iloc[i]/12
            age_rescaled = int(age)#//12 
            
            if 'test_cancer' in split:
                list_test_cancer.append([input_image_path,int(age)])
            
            elif 'train' in split and len(age_bins[age_rescaled])<=n_per_age:
                list_train.append([input_image_path,age])
                counter=counter+1
                y_train.append(age_rescaled)
                age_bins[age_rescaled].append(counter)
                y_train_strat.append(age_rescaled)
                
            elif ('test_healthy' in split) or ('unsupervised' in split) or ('external_test'==split):
                list_test_healthy.append([input_image_path,(age)])
                #print(split)
                
            elif 'test_long' in split or 'test2' in split  : 
                list_test_long.append([input_image_path,(age)])    
            
            else:
                print("Error: ", split)
                
    return list_train, list_test_healthy, list_test_cancer,list_test_long,y_train,age_bins,y_train_strat

list_train, list_test_healthy, list_test_cancer,list_test_long,y_train,age_bins,y_train_strat = create_dataset(df)

print( "test_long: ", len(list_test_long))
'''
# split train and val
train_idx, val_idx = train_test_split(np.arange(len(list_train)),test_size=0.3, shuffle=True, stratify=y_train_strat)
train_list, val_list = [],[]
for i in train_idx:
    train_list.append(list_train[i])
    
for i in val_idx:
    val_list.append(list_train[i]) 
    
#use old train_idx and val_idx from files
df_list_train = pd.DataFrame(np.asarray(list_train))
df_list_train['path_only'] = df_list_train[0].str.split("/",-1).str[-1]
train_idx = pd.read_csv("/media/sda/Anna/brain_age/DiffMIC/dataset/brains_"+str(centile)+"/train_balanced256_"+str(centile)+"_years.csv")
val_idx = pd.read_csv("/media/sda/Anna/brain_age/DiffMIC/dataset/brains_"+str(centile)+"/val_balanced256_"+str(centile)+"_years.csv")
train_list, val_list = [],[]
#print(df_list_train)
for i in range(len(train_idx)):
    old_path = train_idx['dir'].iloc[i].split("/")[-1]
    train_list.append([train_idx['dir'].iloc[i],df_list_train[df_list_train['path_only']==old_path][1].iloc[0]])

for i in range(len(val_idx)):
    old_path = val_idx['dir'].iloc[i]
    val_list.append([val_idx['dir'].iloc[i],df_list_train[df_list_train[0]==old_path][1].iloc[0]])
'''
#print(len(train_list),len(val_list),len(list_test_healthy),len(list_test_cancer),len(list_test_long))

#save train_list into a csv file
if os.path.exists("dataset/brains_"+str(centile))==False:
    os.mkdir("dataset/brains_"+str(centile))
    
#pd.DataFrame(np.asarray(train_list)).to_csv("dataset/brains_"+str(centile)+"/train_balanced256_"+str(centile)+"_years.csv", header=['dir','label'], index=None)
#pd.DataFrame(np.asarray(val_list)).to_csv("dataset/brains_"+str(centile)+"/val_balanced256_"+str(centile)+"_years.csv", header=['dir','label'], index=None)

#pd.DataFrame(np.asarray(list_test_healthy)).to_csv("dataset/brains_"+str(centile)+"/test_healthy256_"+str(centile)+"_years.csv", header=['dir','label'], index=None)
#pd.DataFrame(np.asarray(list_test_cancer)).to_csv("dataset/brains_"+str(centile)+"/test_cancer256"+str(centile)+"_years.csv", header=['dir','label'], index=None)
pd.DataFrame(np.asarray(list_test_long)).to_csv("dataset/brains_"+str(centile)+"/test_external256"+str(centile)+"_years.csv", header=['dir','label'], index=None)
#pd.DataFrame(np.asarray(list_external_test)).to_csv("dataset/brains_"+str(centile)+"/test_external256"+str(centile)+"_years.csv", header=['dir','label'], index=None)

#/media/sda/Anna/brain_age/pytorch/data/2d_256/train/sub-0370_run-1_T1w_111.png