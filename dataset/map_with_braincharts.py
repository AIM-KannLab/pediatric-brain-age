import os 
import numpy as np
import pandas as pd
import statistics

path_to_brainchart = '/media/sda/Anna/brain_age/DiffMIC/dataset/csv_data/allnew_AIM_.csv'
path_to_test_1 = '/media/sda/Anna/brain_age/DiffMIC/notebooks/5models_combined_results.csv'
path_to_test_2 = '/media/sda/Anna/brain_age/DiffMIC/notebooks/brain_age_ext_test2.csv'
path_to_abcd = '/media/sda/Anna/brain_age/DiffMIC/dataset/csv_data/abcd_global_long_AIM_wCent.csv'

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

    return np.average(good_nums_avg)

df_brainchart = pd.read_csv(path_to_brainchart)
df_test_1 = pd.read_csv(path_to_test_1)
df_test_2 = pd.read_csv(path_to_test_2)
df_abcd_long = pd.read_csv(path_to_abcd)
# there needs to be unique mapper for each dataset
##BABY - no matches?
df_baby_brainage = df_test_1[df_test_1['dataset'] == 'BABY']
df_baby_charts = df_brainchart[df_brainchart['study'] == 'BabyConnectome']

df_raw_baby = pd.read_csv("/mnt/kannlab_rfa/Anna/data_baby_connectome/image03/botht1andt2.csv",delimiter=",",header=0)
df_raw_baby['key'] = df_raw_baby['IMAGE_FILE'].str.split("/",-1).str[-1].str.split(".",-1).str[0]
df_raw_baby['SRC_SUBJECT_ID'] = df_raw_baby['SRC_SUBJECT_ID'].astype('string').str.split(".", -1).str[0]
df_raw_baby=df_raw_baby[['SRC_SUBJECT_ID','key']]

df_baby_brainage['key'] = df_baby_brainage['path'].str.split("/",-1).str[-1].str.split(".",-1).str[0].str.split("_",-1).str[0:-1].str.join("_")
merged_baby = pd.merge(df_baby_brainage, df_raw_baby, on=['key'])

df_baby_charts['SRC_SUBJECT_ID'] =  df_baby_charts['participant'].str.split("-",-1).str[1].astype('string')
print(pd.merge(merged_baby, df_baby_charts, on=['SRC_SUBJECT_ID']).columns)
merged_baby_all = pd.merge(merged_baby, df_baby_charts, on=['SRC_SUBJECT_ID'])[['gt','25','37','50','62','75','dataset', 'age_days', 'sex_x',
                                                                      "GMV_cent","WMV_cent","sGMV_cent","CSF_cent",#"TCV_cent","CT_cent","SA_cent",
                                                                      "GMV","WMV","sGMV","Ventricles",#"TCV","CT","SA",
                                                                      'key','path']]

## IXI
df_ixi_brainage = df_test_1[df_test_1['dataset'] == 'IXI']
df_ixi_charts = df_brainchart[(df_brainchart['study'] == 'IXI')]

df_ixi_brainage['key'] = df_ixi_brainage['path'].str.split("/",-1).str[-1].str.split("-",-1).str[0].str.replace("IXI","").astype('int')
df_ixi_charts['key']=df_ixi_charts['participant'].str.split("-", -1).str[1].astype('int')
merged_ixi = pd.merge(df_ixi_brainage, df_ixi_charts, on=['key'])[['gt','25','37','50','62','75','dataset', 'age_days',  'sex_x',
                                                                      "GMV_cent","WMV_cent","sGMV_cent","CSF_cent",#"TCV_cent","CT_cent","SA_cent",
                                                                      "GMV","WMV","sGMV","Ventricles",#"TCV","CT","SA",
                                                                      'key','path']]


## Pixar
df_pixar_brainage = df_test_1[df_test_1['dataset'] == 'Pixar']
df_pixar_charts = df_brainchart[(df_brainchart['study'] == 'Pixar')]

df_pixar_brainage['key']=df_pixar_brainage['path'].str.split("/",-1).str[-1].str.split("_",-1).str[0]
df_pixar_charts['key']=df_pixar_charts['participant']

merged_pixar = pd.merge(df_pixar_brainage, df_pixar_charts, on=['key'])[['gt','25','37','50','62','75','dataset', 'age_days',  'sex_x',
                                                                     "GMV_cent","WMV_cent","sGMV_cent","CSF_cent",#"TCV_cent","CT_cent","SA_cent",
                                                                     "GMV","WMV","sGMV","Ventricles",#"TCV","CT","SA",
                                                                      'key','path']]

## SALD
df_sald_brainage = df_test_1[df_test_1['dataset'] == 'SALD']
df_sald_charts = df_brainchart[(df_brainchart['study'] == 'SALD')]

df_sald_brainage['key']=df_sald_brainage['path'].str.split("/",-1).str[-1].str.split("_",-1).str[0]
df_sald_charts['key']=df_sald_charts['participant']
merged_sald = pd.merge(df_sald_brainage, df_sald_charts, on=['key'])[['gt','25','37','50','62','75','dataset',  'age_days',  'sex_x',
                                                                      "GMV_cent","WMV_cent","sGMV_cent","CSF_cent",#"TCV_cent","CT_cent","SA_cent",
                                                                      "GMV","WMV","sGMV","Ventricles",#"TCV","CT","SA",
                                                                      'key','path']]

## abcd
df_abcd_brainage = df_test_2[(df_test_2['dataset'] == 'ABCD') | (df_test_2['dataset'] == 'ABCD_2023')]
#df_abcd_charts = df_brainchart[(df_brainchart['study'] == 'ABCD')]
df_abcd_charts=df_abcd_long
#sub-NDARINV769BNJY5_ses-baselineYear1Arm1_run-01_T1w_129.png
# filter only those that have 'baseline' in the path
#df_abcd_brainage = df_abcd_brainage[df_abcd_brainage['path'].str.contains("baseline")]
#sub-NDARINV003RTV85_baseline_year_1_arm_1
df_abcd_brainage['key']=df_abcd_brainage['path'].str.split("/",-1).str[-1].str.split("_",-1).str[0:-3].str.join("_").str.replace("sub-","").str.replace("ses-","").str.lower().str.replace("_","")
df_abcd_charts['key']=df_abcd_charts['participant'].str.replace("_","").str.replace("sub-","").str.lower() #sub-NDARINV003RTV85_baseline_year_1_arm_1
#['key']=df_abcd_charts['participant']
#rename Ventricles_cent to CSF_cent
print(df_abcd_charts.columns)
df_abcd_charts.rename(columns={'Ventricles_cent':'CSF_cent'}, inplace=True)

merged_abcd = pd.merge(df_abcd_brainage, df_abcd_charts, on=['key'])[['gt','25','37','50','62','75','dataset',  'age_days',  'sex',
                                                                      "GMV_cent","WMV_cent","sGMV_cent","CSF_cent",#"TCV_cent","CT_cent","SA_cent",
                                                                      "GMV","WMV","sGMV","Ventricles",#"TCV","CT","SA",
                                                                      'key','path']]



print(merged_abcd)
## wu1200
df_wu_brainage = df_test_2[(df_test_2['dataset'] == 'wu1200')]
df_wu_charts = df_brainchart[(df_brainchart['study'] == 'HCP')]

#/media/sda/Anna/brain_age/pytorch/preprocessed_T1/test2/sub-NDARINV769BNJY5_ses-baselineYear1Arm1_run-01_T1w_129.png
# filter only those that have 'baseline' in the path
df_wu_brainage['key']=df_wu_brainage['path'].str.split("/",-1).str[-1].str.split("_",-1).str[0]
df_wu_charts['key']=df_wu_charts['participant'].str.replace("sub-","")

merged_wu = pd.merge(df_wu_brainage, df_wu_charts, on=['key'])[['gt','25','37','50','62','75','dataset',  'age_days',  'sex',
                                                                      "GMV_cent","WMV_cent","sGMV_cent","CSF_cent",#"TCV_cent","CT_cent","SA_cent",
                                                                      "GMV","WMV","sGMV","Ventricles",#"TCV","CT","SA",
                                                                      'key','path']]


# append merged_abcd to merged_sald by columns 25,37,50,62,75,dataset, sex, age_days, GMV	WMV	sGMV	CSF	TCV	CT	SA, key,path
#append merged_baby_all to merged_abcd, merged_pixar
total_df = merged_baby_all.append(merged_ixi, ignore_index=True)
total_df = total_df.append(merged_pixar, ignore_index=True)
total_df = total_df.append(merged_sald, ignore_index=True)
total_df = total_df.append(merged_wu, ignore_index=True)

total_df = total_df.append(merged_abcd)
#apply outlier_voting to columns 25,37,50,62,75
total_df['pred_age_no_outlier']= total_df[['25','37','50','62','75']].apply(outlier_voting, axis=1)
total_df['age_delta'] = total_df['pred_age_no_outlier']-total_df['age_days']/365
total_df['age_years'] = (total_df['age_days']/365).astype(int)
total_df.to_csv("/media/sda/Anna/brain_age/DiffMIC/notebooks/brain_charts.csv", index=False)

