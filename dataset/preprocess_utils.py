import sys
sys.path.append('../TM2_segmentation')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import logging
import SimpleITK as sitk
from scipy.signal import medfilt
import numpy as np
import nibabel as nib
import scipy
import skimage
import functools
from skimage.transform import resize
import subprocess
import pandas as pd
import shutil
import itk

# compute the intersection over union of two binary masks
def iou(component1, component2):
    component1 = np.array(component1, dtype=bool)
    component2 = np.array(component2, dtype=bool)

    overlap = component1 * component2 # Logical AND
    union = component1 + component2 # Logical OR

    IOU = overlap.sum()/float(union.sum())
    return IOU

# helper function to get the id and path of the image and the mask
def get_id_and_path(row, image_dir, nested = False, no_tms=True):
    patient_id, image_path, ltm_file, rtm_file = "","","",""
    if no_tms and row['Ok registered? Y/N'] == "N" :
        print("skip - bad registration")
        return "","","",""
    if "NDAR" in str(row['Filename']) and nested==False and no_tms:
        patient_id = str(row['Filename']).split("_")[0]
    else:
        patient_id = str(row['Filename']).split(".")[0]

    path = find_file_in_path(patient_id, os.listdir(image_dir))
    
    if nested:
        patient_id =  patient_id.split("/")[-1]
        path = patient_id.split("/")[-1]
    if no_tms==False:
        path=""
        
    scan_folder = image_dir+path
    patient_id=patient_id.split("/")[-1]
    
    for file in os.listdir(scan_folder):
        t = image_dir+path+"/"+file
        if "LTM" in file:
            ltm_file = t
        elif "RTM" in file:
            rtm_file = t
        elif "TM" in file:
            rtm_file = t
            ltm_file = t
        if patient_id in file:
            image_path = t
    return patient_id, image_path, ltm_file, rtm_file

# another helper function to get the id and path of the image and the mask, when the folder structure is different
def get_id_and_path_not_nested(row, image_dir, masks_dir):
    patient_id, image_path, tm_file = 0,0,0
    if row['Ok registered? Y/N'] == "N":
        print("skip - bad registration")
        return 0,0,0,0
    if "NDAR" in row['Filename']:
        patient_id = row['Filename'].split("_")[0]
    else:
        patient_id = row['Filename'].split(".")[0]

    path = find_file_in_path(patient_id, os.listdir(masks_dir))
    if len(path)<3:
        return 0,0,0,0
    scan_folder_masks = masks_dir+path

    for file in os.listdir(scan_folder_masks):
        if "._" in file: #skip hidden files
            continue
        if "TM" in file:
            tm_file = masks_dir+path+"/"+file
        elif ".nii" in file and "TM" not in file:
            image_path = image_dir+patient_id+".nii"

    return patient_id, image_path, tm_file     

# crop the image to the bounding box of the mask
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# find the file in the path
def find_file_in_path(name, path):
    result = []
    result = list(filter(lambda x:name in x, path))
    if len(result) != 0:
        for file in result:
            if "._" in file:#skip hidden files
                continue
            else:
                return file
    else:
        return ""

# perform the bias field correction
def bias_field_correction(img):
    image = sitk.GetImageFromArray(img)
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4

    corrector.SetMaximumNumberOfIterations([100] * numberFittingLevels)
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected_image_full_resolution = image / sitk.Exp(log_bias_field)
    return sitk.GetArrayFromImage(corrected_image_full_resolution)

def load_nii(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine

def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return

def denoise(volume, kernel_size=3):
    return medfilt(volume, kernel_size)
  
# apply the windowing to the image   
def apply_window(image, win_centre= 40, win_width= 400):
    range_bottom = 149 #win_centre - win_width / 2
    scale = 256 / 256 #win_width
    image = image - range_bottom

    image = image * scale
    image[image < 0] = 0
    image[image > 255] = 255
    return image

# rescale the intensity of the image and binning
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

# equalize the histogram of the image
def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume

# enhance the image
def enhance(volume, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    try:
        volume = bias_field_correction(volume)
        volume = denoise(volume, kernel_size)
        volume = rescale_intensity(volume, percentils, bins_num)
        if eh:
            volume = equalize_hist(volume, bins_num)
        return volume
    except RuntimeError:
        logging.warning('Failed enchancing')

# enhance the image without bias field correction
def enhance_noN4(volume, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    try:
        #volume = bias_field_correction(volume)
        volume = denoise(volume, kernel_size)
        #print(np.shape(volume))
        volume = rescale_intensity(volume, percentils, bins_num)
        #print(np.shape(volume))
        if eh:
            volume = equalize_hist(volume, bins_num)
        return volume
    except RuntimeError:
        logging.warning('Failed enchancing')

# get the resampled image
def get_resampled_sitk(data_sitk,target_spacing):
    new_spacing = target_spacing

    orig_spacing = data_sitk.GetSpacing()
    orig_size = data_sitk.GetSize()

    new_size = [int(orig_size[0] * orig_spacing[0] / new_spacing[0]),
              int(orig_size[1] * orig_spacing[1] / new_spacing[1]),
              int(orig_size[2] * orig_spacing[2] / new_spacing[2])]

    res_filter = sitk.ResampleImageFilter()
    img_sitk = res_filter.Execute(data_sitk,
                                new_size,
                                sitk.Transform(),
                                sitk.sitkLinear,
                                data_sitk.GetOrigin(),
                                new_spacing,
                                data_sitk.GetDirection(),
                                0,
                                data_sitk.GetPixelIDValue())

    return img_sitk

# convert the nrrd file to nifty file  
def nrrd_to_nifty(nrrd_file):
    _nrrd = nrrd.read(nrrd_file)
    data_f = _nrrd[0]
    header = _nrrd[1]
    return np.asarray(data_f), header

# crop the brain from the image
def crop_brain(var_img, mni_img):
        # invert brain mask 
        inverted_mask = np.invert(mni_img.astype(bool)).astype(float)
        mask_data = inverted_mask * var_img 
        return mask_data

# normalize the image with the brain mask
def brain_norm_masked(mask_data, brain_data, to_save=False):
    masked = crop_brain(brain_data, mask_data)
    enhanced = enhance(masked)
    return enhanced

# enhance all the images in the path
def enhance_and_debias_all_in_path(image_dir='data/mni_templates_BK/',path_to='data/denoised_mris/',\
    input_annotation_file = 'data/all_metadata.csv'):

    df = pd.read_csv(input_annotation_file,header=0)
    df=df[df['Ok registered? Y/N']=='Y'].reset_index()
    #print(df.shape[0])
    for idx in range(0, 1):
        print(idx)
        row = df.iloc[idx]
        patient_id, image_path, tm_file, _ = get_id_and_path(row, image_dir)
        print(patient_id, image_path, tm_file)
        image_sitk =  sitk.ReadImage(image_path)
        image_array  = sitk.GetArrayFromImage(image_sitk)
        image_array = enhance(image_array) 
        image3 = sitk.GetImageFromArray(image_array)
        sitk.WriteImage(image3,path_to+patient_id+'.nii') 
    return 

# Z-enhance all the images in the path
def z_enhance_and_debias_all_in_path(image_dir='data/mni_templates_BK/',path_to='data/z_scored_mris/',\
    input_annotation_file = 'data/all_metadata.csv', for_training=True, annotations=True):
    df = pd.read_csv(input_annotation_file,header=0)
    
    if for_training:
        df=df[df['Ok registered? Y/N']=='Y'].reset_index()
    print(df.shape[0])
    
    for idx in range(0, df.shape[0]):
        print(idx)
        row = df.iloc[idx]
        patient_id, image_path, tm_file, _ = get_id_and_path(row, image_dir, nested=False, no_tms=for_training)
        print(patient_id, len(image_path), tm_file, path_to)
        if not os.path.isdir(path_to+"no_z"):
            os.mkdir(path_to+"no_z")
        if not os.path.isdir(path_to+"z"):
            os.mkdir(path_to+"z")
            
        if len(image_path)>3:
            image_sitk =  sitk.ReadImage(image_path)
            image_array  = sitk.GetArrayFromImage(image_sitk)
            print(len(image_array))
            try:
                image_array = enhance_noN4(image_array)
                image3 = sitk.GetImageFromArray(image_array)
                sitk.WriteImage(image3,path_to+"no_z/"+patient_id+'.nii') 
                os.mkdir(path_to+"z/"+patient_id)
                if annotations:
                    shutil.copyfile(tm_file, path_to+"z/"+patient_id+"/TM.nii.gz")
                duck_line = "zscore-normalize "+path_to+"no_z/"+patient_id+".nii -o "+path_to+"z/"+patient_id +"/"+patient_id+'.nii'
                subprocess.getoutput(duck_line)
            except:
                continue

# find the closest value in the list
def closest_value(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i], i

# find the centile of the input value
def find_centile(input_tmt, age, df):
    #print("TMT:",input_tmt,"Age:", age)
    val,i=closest_value(df['x'],age)
    
    centile = 'out of range'
    if input_tmt<df.iloc[i]['X3']:
        centile ='< 3'
    if df.iloc[i]['X3']<=input_tmt<df.iloc[i]['X10']:
        centile ='3-10'
    if df.iloc[i]['X10']<=input_tmt<df.iloc[i]['X25']:
        centile ='10-25'
    if df.iloc[i]['X25']<=input_tmt<df.iloc[i]['X50']:
        centile ='25-50'
    if df.iloc[i]['X50']<=input_tmt<df.iloc[i]['X75']:
        centile ='50-75'
    if df.iloc[i]['X75']<=input_tmt<df.iloc[i]['X90']:
        centile ='75-90'
    if df.iloc[i]['X90']<=input_tmt<df.iloc[i]['X97']:
        centile ='90-97'
    if input_tmt>df.iloc[i]['X97']:
        centile ='97>'
    #print(val,i,centile)
    return centile

# find the exact percentile of the input value
def find_exact_percentile_return_number(input_tmt, age, df):
    #print("TMT:",input_tmt,"Age:", age)
    val,i=closest_value(df['x'],age)
    
    mu = df.iloc[i]['mu']
    sigma = df.iloc[i]['sigma']
    nu = df.iloc[i]['nu']
    #tau = df.iloc[i]['tau']
    
    if nu!=0:
        z = ((input_tmt/mu)**(nu)-1)/(nu*sigma)
    else:
        z = 1/sigma * math.log(input_tmt/mu)
    percentile = scipy.stats.norm.cdf(z)
    return round(percentile*100,2)

# add median labels to boxplots
def add_median_labels(ax, fmt='.1f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='ultralight', color='gray')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

#register to a template        
def register_to_template(input_image_path, output_path, fixed_image_path,create_subfolder=True):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('data/golden_image/mni_templates/Parameters_Rigid.txt')

    if "nii" in input_image_path and "._" not in input_image_path:
        print(input_image_path)

        # Call registration function
        try:        
            moving_image = itk.imread(input_image_path, itk.F)
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            
            if create_subfolder:
                new_dir = output_path+image_id.split(".")[0] 
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
                itk.imwrite(result_image, new_dir+"/"+image_id)
            else:
                itk.imwrite(result_image, output_path+"/"+image_id)
                
            print("Registered ", image_id)
        except:
            print("Cannot transform", input_image_path.split("/")[-1])
        
if __name__=="__main__":
    # replace header with ,AGE_M,SEX,SCAN_PATH,Filename,dataset
    '''
    z_enhance_and_debias_all_in_path(image_dir='data/mni_templates_BK/',
                                    path_to='data/z_scored_mris/z_with_pseudo/',\
                                    input_annotation_file = 'data/all_metadata.csv')
                                    
    z_enhance_and_debias_all_in_path(image_dir='data/curated_test/reg_tm_not_corrected/',
                                    path_to='data/curated_test/final_test/',
                                     input_annotation_file = 'data/curated_test/reg_tm_not_corrected/Dataset_test_rescaled.csv',
                                     for_training=True)
    # all the datasets
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/registered_not_ench/',
                                     path_to='data/t1_mris/registered/',
                                     input_annotation_file = 'data/Dataset_t1_healthy_raw.csv',
                                     for_training=False,annotations=False)
    
    #ping 
    # z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/pings_registered/',
                                     path_to='data/t1_mris/pings_ench_reg/',
                                     input_annotation_file = 'data/Dataset_ping.csv',
                                     for_training=False, annotations=False)
    # pixar
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/pixar/',
                                     path_to='data/t1_mris/pixar_ench/',
                                     input_annotation_file = 'data/Dataset_pixar.csv',
                                     for_training=False, annotations=False)
    #abide
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/abide_registered/',
                                     path_to='data/t1_mris/abide_ench_reg/',
                                     input_annotation_file = "data/Dataset_abide.csv",
                                     for_training=False, annotations=False)
        
    # calgary
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/calgary_reg/',
                                     path_to='data/t1_mris/calgary_reg_ench/',
                                     input_annotation_file = "data/Dataset_calgary.csv",
                                     for_training=False, annotations=False)       
                                                             
    # aomic replace header with ,AGE_M,SEX,SCAN_PATH,Filename,dataset
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/aomic_reg/',
                                     path_to='data/t1_mris/aomic_reg_ench/',
                                     input_annotation_file = "data/Dataset_aomic.csv",
                                     for_training=False, annotations=False)
                                                                    
    # NIHM replace header with ,AGE_M,SEX,SCAN_PATH,Filename,dataset
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/nihm_reg/',
                                     path_to='data/t1_mris/nihm_ench_reg/',
                                     input_annotation_file = "data/Dataset_nihm.csv",
                                    for_training=False, annotations=False)
                                     
    # ICBM replace header with ,AGE_M,SEX,SCAN_PATH,Filename,dataset
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/icbm_reg/',
                                     path_to='data/t1_mris/icbm_ench_reg/',
                                     input_annotation_file = "data/Dataset_icbm.csv",
                                     for_training=False, annotations=False)                             

    # SALD
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/sald_reg/',
                                     path_to='data/t1_mris/sald_reg_ench/',
                                     input_annotation_file = "data/Dataset_sald.csv",
                                     for_training=False, annotations=False)
    
    ## NYU
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/nyu_reg/',
                                     path_to='data/t1_mris/nyu_reg_ench/',
                                     input_annotation_file = "data/Dataset_nyu.csv",
                                     for_training=False, annotations=False)
    ## NAH
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/healthy_adults_nihm/',
                                     path_to='data/t1_mris/healthy_adults_nihm_reg_ench/',
                                     input_annotation_file = "data/Dataset_healthy_adults_nihm.csv",
                                     for_training=False, annotations=False)
    ## Petfrog
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/petfrog_reg/',
                                     path_to='data/t1_mris/petfrog_reg_ench/',
                                     input_annotation_file = "data/Dataset_petfrog.csv",
                                     for_training=False, annotations=False)
    ## CBTN
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/cbtn_reg/',
                                     path_to='data/t1_mris/cbtn_reg_ench/',
                                     input_annotation_file = "data/Dataset_cbtn.csv",
                                     for_training=False, annotations=False)                              
    ## DMG
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/dmg_reg/',
                                     path_to='data/t1_mris/dmg_reg_ench/',
                                     input_annotation_file = "data/Dataset_dmg.csv",
                                     for_training=False, annotations=False)
    ## BCH
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/bch_reg/',
                                     path_to='data/t1_mris/bch_reg_ench/',
                                     input_annotation_file = "data/Dataset_bch.csv",
                                     for_training=False, annotations=False)
    ## BCH long
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/bch_long_reg/',
                                     path_to='data/t1_mris/bch_long_reg_ench/',
                                     input_annotation_file = "data/Dataset_bch_long.csv",
                                     for_training=False, annotations=False)
    ## 28
    z_enhance_and_debias_all_in_path(image_dir='data/t1_mris/uscf_reg/',
                                     path_to='data/t1_mris/uscf_reg_ench/',
                                     input_annotation_file = "data/Dataset_ucsf.csv",
                                     for_training=False, annotations=False)'''
    ## BCH long masked test
    z_enhance_and_debias_all_in_path(image_dir='data/bch_long_pre_test/reg/',
                                     path_to='data/bch_long_pre_test/reg_ench/',
                                     input_annotation_file = "data/Dataset_bch_long_pre_test.csv",
                                     for_training=False, annotations=False)
    
    