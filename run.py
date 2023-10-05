from __future__ import generators

import logging
import glob, os, functools
import sys
sys.path.append('../')
sys.path.append("./HDBET/")
import argparse
import SimpleITK as sitk
import numpy as np
import scipy
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage
from skimage.transform import resize,rescale
import cv2
import itk
import subprocess

import pandas as pd
import warnings
import statistics

import csv
import os
import yaml

from HDBET.HD_BET.run import run_hd_bet # git clone HDBET repo
from dataset.preprocess_utils import enhance, enhance_noN4
from dataset.preprocess_datasets_T1_to_2d import create_quantile_from_brain

warnings.filterwarnings('ignore')
cuda_device = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

img_path = '../example_data/sub-pixar015_T1w.nii.gz'
path_to = "../output/" # save to

# MNI templates http://nist.mni.mcgill.ca/pediatric-atlases-4-5-18-5y/
age_ranges = {"../example_data/mni_templates/nihpd_asym_04.5-08.5_t1w.nii" : {"min_age":3, "max_age":7},
                "../example_data/mni_templates/nihpd_asym_07.5-13.5_t1w.nii": {"min_age":8, "max_age":13},
                "../example_data/mni_templates/nihpd_asym_13.0-18.5_t1w.nii": {"min_age":14, "max_age":35}}


def select_template_based_on_age(age):
    for golden_file_path, age_values in age_ranges.items():
        if age_values['min_age'] <= int(age) and int(age) <= age_values['max_age']: 
            print(golden_file_path)
            return golden_file_path
        
def register_to_template(input_image_path, output_path, fixed_image_path,rename_id,create_subfolder=True):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('../example_data/mni_templates/Parameters_Rigid.txt')

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
            
            itk.imwrite(result_image, output_path+"/"+rename_id+".nii.gz")
                
            print("Registered ", rename_id)
        except:
            print("Cannot transform", rename_id)
            
def outlier_voting(numbers):
    mean = statistics.mean(numbers)
    stdev = statistics.stdev(numbers)

    threshold = stdev # *2  #*3
    
    good_nums_avg =[]
    for n in numbers:
        if n > mean + threshold or n < mean - threshold:
            continue
        else:
            good_nums_avg.append(n)
    
    #if len(good_nums_avg)<=3:
    #    print(len(good_nums_avg))
    return np.average(good_nums_avg)

if __name__ == '__main__':
    #todo: create option for batched processing of inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='example_data')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--chronological_age', type=float)

    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_path = args.output_path
    gt_age = args.chronological_age # maybe make a list instead for multiple ages and multiple images
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.CUDA_VISIBLE_DEVICES)
    
    proj_dir = './'
    
    ## load image
    for img_path in glob.glob(input_dir+"/*.nii.gz"):
        print("Processing:", img_path)
        nii= nib.load(img_path)
        image, affine  = nii.get_fdata(), nii.affine

        # path to store registered image in
        new_path_to = output_path+"/"+img_path.split("/")[-1].split(".")[0]
        if not os.path.exists(path_to):
            os.mkdir(path_to)
        if not os.path.exists(new_path_to):
            os.mkdir(new_path_to)

        # register image to MNI template
        golden_file_path = select_template_based_on_age(gt_age)
        print("Registering to template:", golden_file_path)
        #fun fact: the registering to the template pipeline is not deterministic
        register_to_template(img_path, new_path_to, golden_file_path,"registered.nii.gz", create_subfolder=False)
            
        image_sitk =  sitk.ReadImage(new_path_to+"/registered.nii.gz")
        image_array  = sitk.GetArrayFromImage(image_sitk)
        image_array = enhance(image_array) # or enhance_noN4(image_array) if no bias field correction is needed
        image3 = sitk.GetImageFromArray(image_array)
        sitk.WriteImage(image3,new_path_to+"/registered_no_z.nii") 

        #skull strip
        run_hd_bet(new_path_to+"/registered_no_z.nii",new_path_to+"/registered_skull_stripped.nii",
                    mode="accurate", 
                    config_file='HDBET/HD_BET/config.py',
                    device=int(cuda_device),
                    postprocess=False,
                    do_tta=True,
                    keep_mask=True, 
                    overwrite=True)

        file_path = new_path_to+"/registered_skull_stripped.nii"
        file_path_mask = new_path_to+"/registered_skull_strip_mask.nii.gz"
        im = nib.load(file_path).get_fdata()
        # rescale pixel values to the range [0, 1]
        im = im + (-im.min())
        #reuse the bet mask to remove the background and the skull
        im_mask = nib.load(file_path_mask).get_fdata()
        im = im * im_mask

        #resample 3d to 2d slices
        img_name = img_path.split("/")[-1].split(".")[0]
        # create slices
        filepaths_list = create_quantile_from_brain(im,img_name,new_path_to,percent_min_present=0.01,save_image=True)
        new_filepath_25, new_filepath_37, new_filepath_50, new_filepath_62, new_filepath_75=filepaths_list
        centile_lst = ['25','37','50','62','75']

        # create .csv for each file and configs
        for i in range(0,len(filepaths_list)):
            filepath=filepaths_list[i]
            csv_path = os.path.join(new_path_to, centile_lst[i] + '.csv')
            
            with open(csv_path, 'w') as f_out:
                writer = csv.writer(f_out)
                writer.writerow(['dir', 'label']) # write header
                writer.writerow([filepath, gt_age])
                #print([filepath, gt_age])
                
            config = 'example_data/config.yml'
            
            with open(config) as f:
                configs = config = yaml.unsafe_load(f)

            configs.data.testdata = new_path_to+"/"+centile_lst[i] + '.csv'
            configs.data.std = 8.8
            with open(new_path_to+'/configs'+centile_lst[i]+'.yml', 'w') as f:
                yaml.dump(configs, f)
                
        age_preds = []
        for centile in centile_lst:
            doc_string = 'brain'
            model_path = 'model_weights/'+str(centile)+'ckpt_best.pth'
            model_path_aux = 'model_weights/'+str(centile)+'aux_ckpt.pth'

            # Open file for writing 
            try:
                os.mkdir(new_path_to+"/logs")
                os.mkdir(new_path_to+"/logs/"+doc_string)
                os.mkdir(new_path_to+"/logs/"+doc_string+"/split_0")
                file = open(new_path_to+"/logs/brain/split_0/testmetrics.txt", "w") 
                file.close()
            except:
                pass

            command = ["python", "main.py",
                    "--device", cuda_device,
                    "--config", new_path_to+'/configs'+centile+'.yml', 
                    "--exp", new_path_to,
                    "--doc", doc_string,
                    "--test","--eval_best",
                    '--eval_path', model_path,
                    '--eval_path_aux',model_path_aux]

            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = result.stdout
            try:
                error = result.stder
            except:
                error = 0
                
            age = float(output.split("\n")[1].split(" ")[-1].replace("[","").replace("]",""))
            age_preds.append(age)
            
        pred_age=outlier_voting(age_preds)
        print("Predicted age:", round(pred_age,2),'years')
        print("MAE:",round(pred_age - gt_age,2),'years')
        
#python main.py --device 0 --config output/sub-pixar015_T1w/configs25.yml --exp output/sub-pixar015_T1w --doc brain --test --eval_best --eval_path model_weights/25ckpt_best.pth --eval_path_aux model_weights/25aux_ckpt.pth