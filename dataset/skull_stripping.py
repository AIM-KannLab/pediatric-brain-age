import sys
sys.path.insert(1, '/media/sda/Anna/brain_age')

import os
import pandas as pd
from HDBET.HD_BET.run import run_hd_bet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


dataset_dict_metadata = {
    #'wu1200':"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_WU1200.csv",}
    #'ABCD_2023': "/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_abcd_new2023.csv",}
    "LONG579":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_long579.csv",
    "28":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_28.csv"}

'''
dataset_dict_metadata = {
    'BABY':"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_t1_healthy_raw.csv", # special case
    'ABCD': "/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_t1_healthy_raw.csv", # special case
    'ABIDE': "/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_abide.csv",
    'AOMIC': "/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_aomic.csv",    
    'Calgary':"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_calgary.csv",
    'HAN':"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_healthy_adults_nihm.csv",
    'HIMH':"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_nihm.csv",
    'ICBM':"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_icbm.csv",
    "IXI":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_t1_healthy_raw.csv", # special case
    "NYU":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_nyu.csv",
    "PING":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_ping.csv",
    "Pixar":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_pixar.csv",
    "SALD":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_sald.csv",
    "BCH":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_bch.csv",
    "28":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_28.csv",
    "DMG":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_dmg.csv",
    #"LONG579":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_long579.csv",
    "Petfrog":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_petfrog.csv",
    "CBTN":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_cbtn.csv"} 
'''

dataset_dict_skullstrip_paths = {
    'ABCD': '/media/sda/Anna/brain_age/pytorch/data/datasets/t1_healthy/', # special case
    'ABIDE': '/media/sda/Anna/brain_age/pytorch/data/datasets/abide/',
    'AOMIC': '/media/sda/Anna/brain_age/pytorch/data/datasets/aomic/',
    'BABY':'/media/sda/Anna/brain_age/pytorch/data/datasets/baby/', # special case
    'Calgary':"/media/sda/Anna/brain_age/pytorch/data/datasets/calgary/",
    'HAN':"/media/sda/Anna/brain_age/pytorch/data/datasets/HAN/",
    'HIMH':'/media/sda/Anna/brain_age/pytorch/data/datasets/NIHM/',
    'ICBM':"/media/sda/Anna/brain_age/pytorch/data/datasets/icbm/",
    "IXI":'/media/sda/Anna/brain_age/pytorch/data/datasets/IXI/', # special case
    "NYU":"/media/sda/Anna/brain_age/pytorch/data/datasets/NYU/",
    "PING":"/media/sda/Anna/brain_age/pytorch/data/datasets/PING/",
    "Pixar":"/media/sda/Anna/brain_age/pytorch/data/datasets/pixar/",
    "SALD":"/media/sda/Anna/brain_age/pytorch/data/datasets/sald/",
    "BCH":'/media/sda/Anna/brain_age/pytorch/data/datasets/bch/',
    'CBTN':'/media/sda/Anna/brain_age/pytorch/data/datasets/cbtn/',
    "28":"/media/sda/Anna/brain_age/pytorch/data/datasets/28/",
    "DMG":"/media/sda/Anna/brain_age/pytorch/data/datasets/dmg/",
    "Petfrog":"/media/sda/Anna/brain_age/pytorch/data/datasets/petfrog/",
    "LONG579":"/media/sda/Anna/brain_age/pytorch/data/datasets/long579/",
    'ABCD_2023': '/media/sda/Anna/brain_age/pytorch/data/datasets/abcd_2023/',
    "wu1200": "/media/sda/Anna/brain_age/pytorch/data/datasets/wu1200/"
}

dataset_dict_paths = {
    'ABCD': "/media/sda/Anna/TM2_segmentation/data/t1_mris/registered/no_z/",
    'ABIDE': "/media/sda/Anna/TM2_segmentation/data/t1_mris/abide_ench_reg/no_z/",
    'AOMIC':"/media/sda/Anna/TM2_segmentation/data/t1_mris/aomic_reg_ench/no_z/",
    'BABY': "/media/sda/Anna/TM2_segmentation/data/t1_mris/registered/no_z/",
    'Calgary': "/media/sda/Anna/TM2_segmentation/data/t1_mris/calgary_reg_ench/no_z/",
    'ICBM': "/media/sda/Anna/TM2_segmentation/data/t1_mris/icbm_ench_reg/no_z/",
    'IXI':"/media/sda/Anna/TM2_segmentation/data/t1_mris/registered/no_z/",
    'HIMH': "/media/sda/Anna/TM2_segmentation/data/t1_mris/nihm_ench_reg/no_z/",
    'PING': "/media/sda/Anna/TM2_segmentation/data/t1_mris/pings_ench_reg/no_z/",
    'Pixar': '/media/sda/Anna/TM2_segmentation/data/t1_mris/pixar_ench/no_z/',
    'SALD': "/media/sda/Anna/TM2_segmentation/data/t1_mris/sald_reg_ench/no_z/",
    'NYU': "/media/sda/Anna/TM2_segmentation/data/t1_mris/nyu_reg_ench/no_z/",
    'HAN': "/media/sda/Anna/TM2_segmentation/data/t1_mris/healthy_adults_nihm_reg_ench/no_z/",
    'Petfrog':"/media/sda/Anna/TM2_segmentation/data/t1_mris/petfrog_reg_ench/no_z/",
    '28':'/media/sda/Anna/TM2_segmentation/data/t1_mris/28_reg_ench/no_z/',
    "BCH":'/media/sda/Anna/TM2_segmentation/data/t1_mris/bch_reg_ench/no_z/',
    "DMG":'/media/sda/Anna/TM2_segmentation/data/t1_mris/dmg_reg_ench/no_z/',
    "CBTN":'/media/sda/Anna/TM2_segmentation/data/t1_mris/cbtn_reg_ench/no_z/',
    "LONG579":'/media/sda/Anna/TM2_segmentation/data/t1_mris/long579_reg_ench/no_z/',
    "ABCD_2023": "/media/sda/Anna/TM2_segmentation/data/t1_mris/abcd_new2023_reg_ench/no_z/",
    "wu1200": "/media/sda/Anna/TM2_segmentation/data/t1_mris/WU1200_reg_ench/no_z/"
    }


dataset_dict_splits = {
    'ABCD': "train", # special case
    'ABIDE': "train",
    'AOMIC': "train",
    'BABY':"test_healthy", # special case
    'Calgary':"train",
    'HAN':"train",
    'HIMH':"train",
    'ICBM':"train",
    "IXI":"test_healthy", # special case
    "NYU":"test_healthy",
    "PING":"train",
    "Pixar":"test_healthy",
    "SALD":"test_healthy",
    "BCH":"test_cancer_bch",
    "CBTN":"test_cancer_cbtn",
    "DMG":"test_cancer_dmg",
    "Petfrog":"train",
    'LONG579':"test2",
    '28':"test2",
    "ABCD_2023": "test2",
    "wu1200": "test2"
}

dict_output_paths = {
    "train":"/media/sda/Anna/brain_age/pytorch/preprocessed_T1/train",
    "test_healthy":"/media/sda/Anna/brain_age/pytorch/preprocessed_T1/test_healthy",
    "test_cancer_bch":"/media/sda/Anna/brain_age/pytorch/preprocessed_T1/test_cancer_bch",
    'test_cancer_cbtn':"/media/sda/Anna/brain_age/pytorch/preprocessed_T1/test_cancer_cbtn",
    'test_cancer_dmg':"/media/sda/Anna/brain_age/pytorch/preprocessed_T1/test_cancer_dmg",
    "test_long":"/media/sda/Anna/brain_age/pytorch/preprocessed_T1/test_long",
    'test_long_single':"/media/sda/Anna/brain_age/pytorch/preprocessed_T1/test_long_single",
    'test2':'/media/sda/Anna/brain_age/pytorch/preprocessed_T1/test2',
}

#run_hd_bet(input_files, output_files, mode, config_file, device, pp, tta, save_mask, overwrite_existing)

input_files = []
output_files = []
for dataset_label,metadata_file in dataset_dict_metadata.items():
    df = pd.read_csv(metadata_file,delimiter=",",header=0)
    df["dataset"]=df["dataset"].astype('string')
    
    df = df[df['dataset']==dataset_label]
    input_path = dataset_dict_paths[dataset_label]
    output_path = dataset_dict_skullstrip_paths[dataset_label]
    print(dataset_label,output_path,len(df))
    
    for i in range(0,len(df)):
        # could be both - .nii.gz and .nii
        file_path = input_path #+str(df['Filename'].iloc[i]).split(".")[0].split("/")[-1]
        found_file = False
        for ext in ['.nii.gz','.nii']:
            #print(file_path +"/"+str(df['Filename'].iloc[i]).split(".")[0].split("/")[-1] + ext)
            if os.path.exists(file_path +"/"+str(df['Filename'].iloc[i]).split(".")[0].split("/")[-1] + ext):
                file_path = file_path +"/"+str(df['Filename'].iloc[i]).split(".")[0].split("/")[-1] + ext
                found_file=True
                break
        if found_file==False:
            print("File not found: ",file_path)
            continue
                
        input_files.append(file_path)
        output_files.append(output_path +str(df['Filename'].iloc[i]).split(".")[0].split("/")[-1] + ".nii.gz")
        #/media/sda/Anna/TM2_segmentation/data/t1_mris/long579_reg_ench/no_z/sub-5589_ses-7_acq-D1S2_T1w.nii
        #/media/sda/Anna/TM2_segmentation/data/t1_mris/long579_reg_ench/no_z/sub-5589_ses-7_acq-D1S2_T1w/sub-5589_ses-7_acq-D1S2_T1w.nii
        #break
        #print(input_files,output_files)
    try:
        run_hd_bet(input_files,output_files,
               mode="accurate", 
               config_file='/media/sda/Anna/brain_age/HDBET/HD_BET/config.py',
               device=0,
             postprocess=False,
             do_tta=True,
             keep_mask=True, 
             overwrite=True)
    except:
        continue

#def run_hd_bet(mri_fnames, output_fnames, mode="accurate", config_file=os.path.join(HD_BET.__path__[0], "config.py"), device=0,
#               postprocess=False, do_tta=True, keep_mask=True, overwrite=True):
    """

    :param mri_fnames: str or list/tuple of str
    :param output_fnames: str or list/tuple of str. If list: must have the same length as output_fnames
    :param mode: fast or accurate
    :param config_file: config.py
    :param device: either int (for device id) or 'cpu'
    :param postprocess: whether to do postprocessing or not. Postprocessing here consists of simply discarding all
    but the largest predicted connected component. Default False
    :param do_tta: whether to do test time data augmentation by mirroring along all axes. Default: True. If you use
    CPU you may want to turn that off to speed things up
    :return:
    """
#data/t1_mris/icbm_ench_reg/z/UTHC_2009/