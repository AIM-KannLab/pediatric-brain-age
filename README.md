# RegDiff

data prep:
0. intitial preprocess: from the Tmt pipeline
1. skull strip pytorch/skull_stripping.py
2. create quantiles  python dataset/preprocess_datasets_T1_to_2d.py
3. create dataframes python dataset/create_df.py 

```
median: python main.py --device 0 --thread 8 --loss diffmic_conditional --config configs/brain.yml --exp results_brain_reg_50 --doc brain --n_splits 1 
python main.py --device 0 --thread 8 --loss diffmic_conditional --config results_brain_reg_med/logs/ --exp results_brain_reg_med --doc brain --n_splits 1 --test --eval_best

 # median mip: python main.py --device 0 --thread 8 --loss diffmic_conditional --config configs/brain_reg_mip.yml --exp results_brain_medians_mip --doc brain --n_splits 1 
python main.py --device 0 --thread 8 --loss diffmic_conditional --config results_brain_medians_mip/logs/ --exp results_brain_medians_mip --doc brain --n_splits 1 --test --eval_best

 # quantile 25: python main.py --device 0 --thread 8 --loss diffmic_conditional --config configs/brain_reg_25.yml --exp results_brain_reg_25 --doc brain --n_splits 1  
python main.py --device 0 --thread 8 --loss diffmic_conditional --config results_brain_reg_25/logs/ --exp results_brain_reg_25 --doc brain --n_splits 1 --test --eval_best

#37 python main.py --device 0 --thread 8 --loss diffmic_conditional --config configs/brain_reg_37.yml --exp results_brain_reg_37 --doc brain --n_splits 1   
 python main.py --device 0 --thread 8 --loss diffmic_conditional --config results_brain_reg_37_y/logs/ --exp results_brain_reg_37_y --doc brain --n_splits 1  --test --eval_best 

# 62 python main.py --device 0 --thread 8 --loss diffmic_conditional --config configs/brain_reg_62.yml --exp results_brain_reg_62 --doc brain --n_splits 1   
python main.py --device 0 --thread 8 --loss diffmic_conditional --config results_brain_reg_62_y/logs/ --exp results_brain_reg_62_y --doc brain --n_splits 1   --test --eval_best 

# quantile 75: python main.py --device 0 --thread 8 --loss diffmic_conditional --config configs/brain_reg_75.yml --exp results_brain_reg_75 --doc brain --n_splits 1 
python main.py --device 0 --thread 8 --loss diffmic_conditional --config results_brain_reg_75/logs/ --exp results_brain_reg_75 --doc brain --n_splits 1 --test --eval_best 
```

## A Quick Overview 

## Requirements
``conda env create -f environment.yml``

## Run
notebooks/Demo-notebook.ipynb

## Thanks
The method is elaborated in the paper [DiffMIC: Dual-Guidance Diffusion Network for Medical Image Classification](https://arxiv.org/abs/2303.10610).
Code is largely based on [XzwHan/CARD](https://github.com/XzwHan/CARD), [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion), [MedSegDiff](https://github.com/WuJunde/MedSegDiff/tree/master), [nyukat/GMIC](https://github.com/nyukat/GMIC)


## Cite
If you find this code useful, please cite 




