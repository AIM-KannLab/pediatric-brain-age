import skimage.transform as skTrans
import nibabel as nib
import numpy as np
from nibabel.affines import rescale_affine
import os
import pandas as pd
import logging
import SimpleITK as sitk
from scipy.signal import medfilt
import itk
import skimage
import functools
from skimage.transform import resize
import subprocess
import shutil
import gc
import cv2
import matplotlib.pyplot as plt

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

dataset_dict_metadata = {#'ABCD': "/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_t1_healthy_raw.csv", }
  
    'wu1200':"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_WU1200.csv",
    'ABCD_2023': "/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_abcd_new2023.csv",
    "LONG579":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_long579.csv",
    "28":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_28.csv"}
'''
 
    'ABIDE': "/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_abide.csv",
    'AOMIC': "/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_aomic.csv",
    'BABY':"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_t1_healthy_raw.csv", # special case
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
    "Petfrog":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_petfrog.csv",
    "CBTN":"/media/sda/Anna/brain_age/pytorch/data/datasets/Dataset_cbtn.csv"}
'''
# bch is incorrect!!!

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
    "LONG579":"/media/sda/Anna/brain_age/pytorch/data/datasets/long579/",
    "Petfrog":"/media/sda/Anna/brain_age/pytorch/data/datasets/petfrog/", 
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
    "BCH":'/media/sda/Anna/TM2_segmentation/data/t1_mris/bch_reg_ench/no_z/',
    "DMG":'/media/sda/Anna/TM2_segmentation/data/t1_mris/dmg_reg_ench/no_z/',
    "CBTN":'/media/sda/Anna/TM2_segmentation/data/t1_mris/cbtn_reg_ench/no_z/', 
    '28':'/media/sda/Anna/TM2_segmentation/data/t1_mris/28_reg_ench/no_z/',
    "LONG579":'/media/sda/Anna/TM2_segmentation/data/t1_mris/long579_reg_ench/no_z/',
    "ABCD_2023": "/media/sda/Anna/TM2_segmentation/data/t1_mris/abcd_new2023_reg_ench/no_z/",
    "wu1200": "/media/sda/Anna/TM2_segmentation/data/t1_mris/WU1200_reg_ench/no_z/"
    }



dataset_dict_splits = {
    'ABCD': "test2", # special case
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
    "train":"/media/sda/Anna/brain_age/pytorch/data/2d_256/train",
    "test_healthy":"/media/sda/Anna/brain_age/pytorch/data/2d_256/test_healthy",
    "test_cancer_bch":"/media/sda/Anna/brain_age/pytorch/data/2d_256/test_cancer_bch",
    'test_cancer_cbtn':"/media/sda/Anna/brain_age/pytorch/data/2d_256/test_cancer_cbtn",
    'test_cancer_dmg':"/media/sda/Anna/brain_age/pytorch/data/2d_256/test_cancer_dmg",
    'test_long_single':"/media/sda/Anna/brain_age/pytorch/data/2d_256/test_long_single",
    'test2':'/media/sda/Anna/brain_age/pytorch/preprocessed_T1/test2',
}

def crop_center(img, cropx,cropy,cropz):
    z,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)  
    startz = z//2-(cropz//2)    
    return img[startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]

def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return

def find_in_metafile(file, df):
    for index, row in df.iterrows():
        if row['filename']==file:
            return row['AGE_M'], row['SEX'], row['dataset']
    print("not found", file)
    not_found.append(file)
    return 0,0,0

def find_file_in_path(name, path):
    result = []
    result = list(filter(lambda x:name in x, path))
    if len(result) != 0:
        return result[0]
    else:
        return False

def register_to_template(input_image_path, fixed_image_path):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('templates/nihpd_obj2_asym_nifti/Parameters_Rigid.txt')

    if ".nii" in input_image_path:
        print(input_image_path)
        # Call registration function
        try:
            moving_image = itk.imread(input_image_path, itk.F)
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            
            #itk.imwrite(result_image, output_path+"/"+image_id)
            print("Registered ", image_id)
            return 0,result_image
        except:
            print("Cannot transform", input_image_path.split("/")[-1])
            return 1,1
   
def select_template_based_on_age(age):
    #https://nist.mni.mcgill.ca/atlases/
    if age<=2*12:
        return '/templates/nihpd_obj2_asym_nifti/nihpd_asym_00-02_t2w.nii'
    if age> 2*12 and age<=5*12:
        return 'templates/nihpd_obj2_asym_nifti/nihpd_asym_02-05_t2w.nii'
    if age> 5*12 and age<=8*12:
        return 'templates/nihpd_obj2_asym_nifti/nihpd_asym_05-08_t2w.nii'
    if age> 8*12 and age<=11*12:
        return 'templates/nihpd_obj2_asym_nifti/nihpd_asym_08-11_t2w.nii'
    if age> 11*12 and age<=14*12:
        return 'templates/nihpd_obj2_asym_nifti/nihpd_asym_11-14_t2w.nii'
    if age> 14*12 and age<=17*12:
        return 'templates/nihpd_obj2_asym_nifti/nihpd_asym_14-17_t2w.nii'
    if age> 17*12 and age<=21*12:
        return 'templates/nihpd_obj2_asym_nifti/nihpd_asym_17-21_t2w.nii'
    if age> 21*12 and age<=27*12:
        return 'templates/nihpd_obj2_asym_nifti/nihpd_asym_21-27_t2w.nii'
    if age> 27*12 and age<=33*12:
        return 'templates/nihpd_obj2_asym_nifti/nihpd_asym_27-33_t2w.nii'
    if age> 33*12:
        return 'templates/nihpd_obj2_asym_nifti/nihpd_asym_33-44_t2w.nii'
    return 0

def load_nii(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine

def denoise(volume, kernel_size=3):
    return medfilt(volume, kernel_size)
    
def apply_window(image, win_centre= 40, win_width= 400):
    range_bottom = 149 #win_centre - win_width / 2
    scale = 256 / 256 #win_width
    image = image - range_bottom

    image = image * scale
    image[image < 0] = 0
    image[image > 255] = 255

    return image

def isNaN(string):
    return string != string

def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    #remove background pixels by the otsu filtering
    t = skimage.filters.threshold_otsu(volume,nbins=6)
    volume[volume < t] = 0
    
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])
    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume

def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume

def __padding_volume__(volume,input_D,input_H,input_W):
        """
        padding the volume to the input size
        """
        [depth, height, width] = volume.shape
        print("volume shape: ", volume.shape)
        padding_depth =input_D - depth
        padding_height = input_H - height
        padding_width = input_W - width
        if padding_depth < 0 or padding_height < 0 or padding_width < 0:
            print("error! invalid input size!")
            exit(0)
        padding_left = int(padding_depth/2)
        padding_right = padding_depth - padding_left
        padding_top = int(padding_height/2)
        padding_bottom = padding_height - padding_top
        padding_front = int(padding_width/2)
        padding_back = padding_width - padding_front
        if len(volume.shape) == 3:
            volume = np.pad(volume, ((padding_left, padding_right),(padding_top, padding_bottom), (padding_front, padding_back)), 'edge')
        return volume
  
def create_quantile_from_brain(im, img_name, output_path, percent_min_present=0.01, save_image=True):
    result = __padding_volume__(im, 256,256,256)
    # get the min box of im_mask - where at least 1% of the pixels are non-zero
    threshold = percent_min_present*np.sum(result.astype(bool))
    valid_range = []
    for k in range(0,256):
        if np.sum(result.astype(bool)[:,:,k])>threshold:
            valid_range.append(k)
            '''img = result[:,:,k]
            # rotate the image by 90 degrees
            img = np.rot90(img, k=1, axes=(0, 1))
            #save the slices
            new_filepath = output_path+"/"+img_name+"_"+str(k)+".png"
            '''
    #find median slice in valid_range
    if len(valid_range)==0:
        print("No valid slices found")
        return 0, 0, 0, 0, 0        
    
            
    quantile_slice25 = int(np.quantile(np.array(valid_range),0.25))
    quantile_slice37 = int(np.quantile(np.array(valid_range),0.375))
    median_slice = int(np.quantile(np.array(valid_range),0.5))
    quantile_slice62 = int(np.quantile(np.array(valid_range),0.625))
    quantile_slice75 = int(np.quantile(np.array(valid_range),0.75))
            
    #25%
    img = result[:,:,quantile_slice25]
    # rotate the image by 90 degrees
    img = np.rot90(img, k=1, axes=(0, 1))
    new_filepath_25 = output_path+"/"+img_name+"_"+str(quantile_slice25)+".png"    
    if save_image: 
        plt.imsave(new_filepath_25, img, cmap='gray')  
    
    #37.5%
    img = result[:,:,quantile_slice37]
    # rotate the image by 90 degrees
    img = np.rot90(img, k=1, axes=(0, 1))
    new_filepath_37 = output_path+"/"+img_name+"_"+str(quantile_slice37)+".png"    
    if save_image: 
        plt.imsave(new_filepath_37, img, cmap='gray')  
    
    #50%
    img = result[:,:,median_slice]
    img = np.rot90(img, k=1, axes=(0, 1))
    new_filepath_50 = output_path+"/"+img_name+"_"+str(median_slice)+".png"    
    if save_image: 
        plt.imsave(new_filepath_50, img, cmap='gray')  
    
    #62.5%
    img = result[:,:,quantile_slice62]
    # rotate the image by 90 degrees
    img = np.rot90(img, k=1, axes=(0, 1))
    new_filepath_62 = output_path+"/"+img_name+"_"+str(quantile_slice62)+".png"    
    if save_image: 
        plt.imsave(new_filepath_62, img, cmap='gray')  
    
    #75%
    img = result[:,:,quantile_slice75]
    # rotate the image by 90 degrees
    img = np.rot90(img, k=1, axes=(0, 1))
    new_filepath_75 = output_path+"/"+img_name+"_"+str(quantile_slice75)+".png"    
    if save_image: 
        plt.imsave(new_filepath_75, img, cmap='gray')  
        
    return new_filepath_25, new_filepath_37, new_filepath_50, new_filepath_62, new_filepath_75        
    
      
def enhance_noN4(volume, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    try:
        volume = denoise(volume, kernel_size)
        volume = rescale_intensity(volume, percentils, bins_num)
        if eh:
            volume = equalize_hist(volume, bins_num)
        return volume
    except RuntimeError:
        logging.warning('Failed enchancing')

if __name__ == '__main__':
    threshold = 4000
    np_subsection,np_subsection_medians = [], []
    np_subsection_25,np_subsection_75 = [], []
    np_subsection_37,np_subsection_62 = [], []
    
    np_subsection_medians_mip = []
    np_subsection_25_mip,np_subsection_75_mip = [], []
    
    j = 0
    fold = 0
    batch_size = 32 #512 #2048
    cubes = []
    
    for dataset_label,metadata_file in dataset_dict_metadata.items():
        df = pd.read_csv(metadata_file,delimiter=",",header=0)
        print(dataset_label)
        df["dataset"]=df["dataset"].astype('string')
        df = df[df['dataset']==dataset_label]
        
        input_path = dataset_dict_paths[dataset_label]
        output_path_split = dataset_dict_splits[dataset_label]
        output_path = dict_output_paths[output_path_split]
        mask_path = dataset_dict_skullstrip_paths[dataset_label]
        print(dataset_label,output_path_split,output_path)

        #fix /media/sda/Anna/TM2_segmentation/data/t1_mris/bch_long_reg_ench/z4315452
        # fix i counter
        
        for i in range(0,len(df)):
            file_path = input_path +str(df['Filename'].iloc[i]).split(".")[0].split("/")[-1] #+\
            file_path_mask = mask_path + str(df['Filename'].iloc[i]).split(".")[0].split("/")[-1] +'_mask.nii.gz'
            
            found_file = False
            
            # could be both - .nii.gz and .nii
            for ext in ['.nii.gz','.nii']:
                if os.path.exists(file_path + ext) and os.path.exists(file_path_mask):
                    file_path = file_path + ext
                    found_file=True
                    break
            if found_file==False:
                print("File not found: ",file_path_mask,os.path.exists(file_path + ext),os.path.exists(file_path_mask))
                continue
                
            print(i, j, file_path)
            if dataset_label=="BABY":
                t1_marker = file_path.split("/")[-1].split("_")[-1]
                if t1_marker=="T1w.nii.gz" or t1_marker=="T1w.nii":
                    n_of_scan = int(file_path.split("/")[-1].split("_")[-2])
                    name_of_scan = "_".join(file_path.split("/")[-1].split("_")[0:-2])
                    find_df = df[df['Filename'].str.contains(str(name_of_scan))]
                    find_last = find_df.iloc[-1]
                    find_df['id'] = find_df['Filename'].str.split("/").str[-1].str.split("_").str[-2].astype(int)
                    n_of_scan_last = int(find_last['Filename'].split("/")[-1].split("_")[-2])
                else:
                    n_of_scan = (file_path.split("/")[-1].split("-")[-2].replace("SE",""))
                    name_of_scan = "-".join(file_path.split("/")[-1].split("-")[0:-2])
                    find_df = df[df['Filename'].str.contains(str(name_of_scan))]
                    find_last = find_df.iloc[-1]
                    find_df['id'] = find_df['Filename'].str.split("/").str[-1].str.split("-").str[-2].str.replace("SE","").astype(int)
                    n_of_scan_last = int(find_last['Filename'].split("/")[-1].split("-")[-2].replace("SE",""))
                
                print(n_of_scan,n_of_scan_last,list(find_df['id']))
                if n_of_scan_last!=n_of_scan:
                    print("Not last scan, skipping")
                    continue
                
            # get the metadata
            age = df['AGE_M'].iloc[i]
            if dataset_label=="DMG" or dataset_label=="BCH":
                age = age*12
            sex = df['SEX'].iloc[i]
            
            im = nib.load(file_path).get_fdata()
            
            # rescale pixel values to the range [0, 1]
            im = im + (-im.min())
            
            #reuse the bet mask to remove the background and the skull
            im_mask = nib.load(file_path_mask).get_fdata()
            im = im * im_mask
            
            # do downsampling
            #rescaled = skTrans.rescale(im, .7, order=1, preserve_range=True)
            
            # crop to 128x128x128
            # result = crop_center(im, 182, 182, 182)
            # result = NormalizeData(result)
            # result = __padding_volume__(im, 256,256,256)
            
            # resample 3d to 2d slices
            img_name = str(df['Filename'].iloc[i]).split(".")[0].split("/")[-1]
            new_filepath_25, new_filepath_37, new_filepath_50, new_filepath_62, new_filepath_75 = create_quantile_from_brain(im, img_name, output_path_split,  percent_min_present=0.01, save_image=False)
            np_subsection_25.append([new_filepath_25,age,sex,output_path_split,dataset_label])  
            np_subsection_37.append([new_filepath_37,age,sex,output_path_split,dataset_label]) 
            np_subsection_medians.append([new_filepath_50,age,sex,output_path_split,dataset_label])
            np_subsection_62.append([new_filepath_62,age,sex,output_path_split,dataset_label])  
            np_subsection_75.append([new_filepath_75,age,sex,output_path_split,dataset_label]) 
                                       
    #df = pd.DataFrame(np_subsection)
    #df.to_csv(path_or_buf="pytorch/data/2D_Dataset_skull_stripped256_.csv", header=False, index=False)
    df = pd.DataFrame(np_subsection_25)
    df.to_csv(path_or_buf="dataset/csv_data/2D_Dataset_skull_stripped25_test2.csv", header=False, index=False)
    
    df = pd.DataFrame(np_subsection_37)
    df.to_csv(path_or_buf="dataset/csv_data/2D_Dataset_skull_stripped37_test2.csv", header=False, index=False)
    
    df = pd.DataFrame(np_subsection_medians)
    df.to_csv(path_or_buf="dataset/csv_data/2D_Dataset_skull_stripped50_test2.csv", header=False, index=False)
    
    df = pd.DataFrame(np_subsection_62)
    df.to_csv(path_or_buf="dataset/csv_data/2D_Dataset_skull_stripped62_test2.csv", header=False, index=False)
    
    df = pd.DataFrame(np_subsection_75)
    df.to_csv(path_or_buf="dataset/csv_data/2D_Dataset_skull_stripped75_test2.csv", header=False, index=False)
  
