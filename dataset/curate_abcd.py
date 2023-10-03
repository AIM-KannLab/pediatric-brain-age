import os
import pandas as pd 
# open file
path_nii1 = '/mnt/kannlab_rfa/Anna/PING/Package_1207895/niftis/IRSPGR_PROMO/'
path_nii2 = '/mnt/kannlab_rfa/Anna/PING/Package_1207895/niftis/MPRAGE/'
input_annotation_file = 'data/csv_files/ping.csv'

import sys
 
# setting path
sys.path.append('../TM2_segmentation')

import os
import numpy as np
import nibabel as nib
import itk
import pandas as pd
import tarfile

from scripts.preprocess_utils import find_file_in_path, register_to_template
from zipfile import ZipFile

# load metadata file     
df = pd.read_csv(input_annotation_file, header=0)
df = df[df['image_description'] == 'MR structural (T1)']
not_found = []

input_path = "/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/t1/"
extracted_path ='/mnt/kannlab_rfa/Anna/PING/Package_1207895/image03/extracted_t1/'

save_to = 'data/t1_mris/pings_registered/'

age_ranges = {  "data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "data/golden_image/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}

path_nii1 = '/mnt/kannlab_rfa/Anna/PING/Package_1207895/niftis/IRSPGR_PROMO'
path_nii2 = '/mnt/kannlab_rfa/Anna/PING/Package_1207895/niftis/MPRAGE'

final_metadata = []
for idx in range(0, df.shape[0]):
    row = df.iloc[idx]
    age = row['interview_age'] 
    sex = row['sex']
    
    res1 = False
    filepath = ""
    for sub1 in os.listdir(path_nii1):
        if row['src_subject_id'] in sub1:
            res1 = True
            filepath = sub1
            input_path = path_nii1
            break
        
    res2 = False
    for sub2 in os.listdir(path_nii2):
        if row['src_subject_id'] in sub2:
            res2=True
            filepath = sub2
            input_path = path_nii2
            break
        
    if res1 or res2:
        # print(row['src_subject_id'])
        # preprocess file
        for golden_file_path, age_values in age_ranges.items():
            if age_values['min_age'] <= age//12 and age//12 <= age_values['max_age']: 
                print(age, input_path+"/"+filepath, save_to, golden_file_path)
                register_to_template(input_path+"/"+filepath, save_to, golden_file_path, create_subfolder=False)
                #0,AGE_M,SEX,SCAN_PATH,Filename,dataset
                final_metadata.append([age,sex,save_to+"/"+filepath,filepath,'PING'])
            # break
       # pass
    else:
        not_found.append(row['src_subject_id'])
    # break
    
df = pd.DataFrame(final_metadata)
df.to_csv(path_or_buf= "data/Dataset_ping.csv")

# print("Not found", len(not_found))
# print(not_found)
# corrupted: ['P0411', 'P0421', 'P0428', 'P0431', 'P0433', 'P0438', 'P0451', 'P0455', 'P0473', 'P0929', 'P1130', 'P1386', 'P1640', 'PING_PG_P1460']
'''for img in not_found:
    for p in os.listdir(input_path):
        if img in p:
            unzip_path = extracted_path+p.split(".")[0] 
            print(unzip_path)
            if not os.path.exists(unzip_path):
                os.mkdir(unzip_path)
                #print(unzip_path)
            
                # loading the temp.zip and creating a zip object
                with ZipFile(input_path+p, 'r') as zObject:
                    zObject.extractall(path=unzip_path)
            else:
                print("skipped ", unzip_path)'''