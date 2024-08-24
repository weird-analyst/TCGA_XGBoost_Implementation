# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:14:05 2024

@author: teesh
"""

import argparse
from numpy.random import seed
import pandas as pd
import random as rn
import os
# import numpy as np
# from sklearn.decomposition import PCA
import time
from preProcess import get_protein, get_mRNA, get_MicroRNA, get_Methylation, \
        standarize_dataset, get_n_years, get_independent_data_single, \
        merge_datasets, normalize_dataset
from classify_all import run_mixture_cv, run_one_race_cv, \
        run_unsupervised_transfer_cv, run_CCSA_transfer, \
        run_supervised_transfer_cv, run_naive_transfer_cv, \
        FADA_classification 
from tensorflow import set_random_seed
from autoencoders import load_or_train_encoders, process_omics

seed(11111)
set_random_seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)

AE_batchsize = 20
AE_iter = 100

folderISAAC = 'GenderBidirectionalTransfer/'
if os.path.exists(folderISAAC)!=True:
    folderISAAC = './'

def main():
    
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process input arguments for ML task.")

    # Adding arguments to the parser
    parser.add_argument(
        "--data_Category", type=str, choices=["GenderRace", "Race", "Gender"], required=True,
        help="Specify the count category: GenderRace, Race, Gender"
    )
    parser.add_argument(
        "--omicsConfiguration", type=str, choices=["single", "combination"], required=True,
        help="Specify if the data is single or combination of omics feature from the TCGA dataset"
    )
    parser.add_argument(
        "--DDP_group", type=str, nargs='?', default=None,
        help="Specify DDP group: BLACK, ASIAN, NAT_A. Required if data_Category is GenderRace or Race"
    )
    parser.add_argument(
        "--cancer_type", type=str,
        help="Cancer Type"
    )
    parser.add_argument(
        "--omics_feature", type=str,
        help="Feature Type. \
        If 'single', then it should be one of the following 'Protein', 'mRNA', 'Methylation', 'MicroRNA'. \
        If 'combination', then it should be given as combination of 2/3/4 features as 'Feature1_'Feature2_Feature3'. "
    )
    parser.add_argument(
        "--endpoint", type=str,
        help="Clinical Outcome Endpoint"
    )
    parser.add_argument(
        "--years", type=int,
        help="Event Time Threshold (years)"
    )
    parser.add_argument(
        "--features_count", type=int, default=200,
        help="No. of Features should be specified only if FeatureMethod value is 0, 1, or 2"
    )
    parser.add_argument(
        "--FeatureMethod", type=int,
        help="0 for pValue, 1 for PCA, 2 for AE, and None for No Feature Selection"
    )
    parser.add_argument(
        "--AutoencoderSettings", type=int, default=1,
        help="1 for L-L-MSE, 2 for R-L-BCE. This is required only if FeatureMethod is 2"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Conditional checks
    if args.data_Category in ["GenderRace", "Race"] and args.DDP_group is None:
        parser.error("DDP_group is required when data_Category is GenderRace or Race.")
    if args.data_Category == "Gender" and args.DDP_group is not None:
        parser.error("DDP_group is not required when data_Category is Gender.")
    if args.FeatureMethod == 2 and args.AutoencoderSettings not in [1, 2]:
        parser.error("AutoencoderSettings must be 1 or 2 when FeatureMethod is 2.")
    if args.FeatureMethod is not None and args.features_count is None:
        parser.error("features_count must be specified when FeatureMethod is 0, 1, or 2.")
        
    # Print parsed arguments for debugging
    print(args, flush=True)
    
    # Assign parsed arguments to variables
    data_Category = args.data_Category
    omicsConfiguration = args.omicsConfiguration
    DDP_group = args.DDP_group if data_Category in ["Race","GenderRace"] else None
    groups = ("WHITE", DDP_group) if data_Category in ["Race","GenderRace"] else ("WHITE","BLACK","ASIAN","NAT_A")
    genders = ("MALE", "FEMALE")
    cancer_type = args.cancer_type
    omics_feature = args.omics_feature
    endpoint = args.endpoint
    years = args.years
    FeatureMethod = args.FeatureMethod
    features_count = args.features_count if FeatureMethod in [0,1,2] else -1
    AutoencoderSettings = args.AutoencoderSettings if FeatureMethod==2 else None
    
    # Debug print to verify correct setup
    print(f"Which count: {data_Category}")
    print(f"Count type: {omicsConfiguration}")
    print(f"DDP group: {DDP_group}")
    print(f"Groups: {groups}")
    print(f"Genders: {genders}")
    print(f"Cancer type: {cancer_type}")
    print(f"Omics feature: {omics_feature}")
    print(f"Endpoint: {endpoint}")
    print(f"Years: {years}")
    print(f"Features count: {features_count}")
    print(f"Feature method: {FeatureMethod}")
    print(f"Autoencoder settings: {AutoencoderSettings}")
    
    if FeatureMethod==0:
        FeatureMethodName = 'ANOVA'
    elif FeatureMethod==1:
        FeatureMethodName = 'PCA'
    elif FeatureMethod==2:
        FeatureMethodName = 'AE'
    else:
        FeatureMethodName = 'OriginalFeatures'
        
    if data_Category=='Gender':
        if FeatureMethod in [0,1]:
            TaskName = 'TCGA-' + data_Category + '-' + \
                        cancer_type + '-' + omics_feature + '-' + endpoint + '-' + str(years) + 'YR' \
                        + '_' + FeatureMethodName + '-' + str(features_count) + 'Features'
        elif FeatureMethod==2:
            TaskName = 'TCGA-' + data_Category + '-' + \
                        cancer_type + '-' + omics_feature + '-' + endpoint + '-' + str(years) + 'YR' \
                        + '_' + FeatureMethodName + '-' + str(AutoencoderSettings) + \
                        '-' + str(features_count) + 'Features'
        else:
            TaskName = 'TCGA-' + data_Category + '-' + \
                        cancer_type + '-' + omics_feature + '-' + endpoint + '-' + str(years) + 'YR' \
                        + '_' + FeatureMethodName
    
    elif data_Category in ['GenderRace', 'Race']:
        if FeatureMethod in [0,1]:
            TaskName = 'TCGA-' + data_Category + '-' + \
                        cancer_type + '-' + groups[1] + '-' + groups[0] + omics_feature + '-' + endpoint + '-' + str(years) + 'YR' \
                        + '_' + FeatureMethodName + '-' + str(features_count) + 'Features'
        elif FeatureMethod==2:
            TaskName = 'TCGA-' + data_Category + '-' + \
                        cancer_type + '-' + groups[1] + '-' + groups[0] + omics_feature + '-' + endpoint + '-' + str(years) + 'YR' \
                        + '_' + FeatureMethodName + '-' + str(AutoencoderSettings) + \
                        '-' + str(features_count) + 'Features'
        else:
            TaskName = 'TCGA-' + data_Category + '-' + \
                        cancer_type + '-' + groups[1] + '-' + groups[0] + omics_feature + '-' + endpoint + '-' + str(years) + 'YR' \
                        + '_' + FeatureMethodName
                        
    print("The ML Task Name is: "+TaskName)
    
    out_file_name = folderISAAC + 'Result/' + TaskName + '.xlsx'
    
    CCSA_path = folderISAAC +'CCSA_data/' + TaskName + '/CCSA_pairs'
    checkpt_path = folderISAAC+'ckpt/FADA_'+TaskName+'_checkpoint.pt'
    
    if os.path.exists(out_file_name):
        print("Result already exists.")
    else:
        if omicsConfiguration=='combination':
            omics_feature = tuple(omics_feature.split('_'))
        
        datasets = {}
        
        # Reading data for input machine learning task
        if isinstance(omics_feature, str):  # single omics
            if omics_feature == 'mRNA':
                datasets['mRNA'] = get_mRNA(cancer_type=cancer_type, endpoint=endpoint, 
                                            groups=groups, genders=genders,
                                            FeatureMethod=FeatureMethod, FeatureTrain=False)
            elif omics_feature == 'MicroRNA':
                datasets['MicroRNA'] = get_MicroRNA(cancer_type=cancer_type, endpoint=endpoint, 
                                            groups=groups, genders=genders, 
                                            FeatureMethod=FeatureMethod, FeatureTrain=False)
            elif omics_feature == 'Protein':
                datasets['Protein'] = get_protein(cancer_type=cancer_type, endpoint=endpoint, 
                                            groups=groups, genders=genders,
                                            FeatureMethod=FeatureMethod, FeatureTrain=False)
            elif omics_feature == 'Methylation':
                datasets['Methylation'] = get_Methylation(cancer_type=cancer_type, endpoint=endpoint, 
                                            groups=groups, genders=genders,
                                            FeatureMethod=FeatureMethod, FeatureTrain=False)
        else:  # multi omics
            for Feature in omics_feature:
                if Feature == 'mRNA':
                    datasets['mRNA'] = get_mRNA(cancer_type=cancer_type, endpoint=endpoint, 
                                            groups=groups, genders=genders,
                                            FeatureMethod=FeatureMethod, FeatureTrain=False)
                elif Feature == 'MicroRNA':
                    datasets['MicroRNA'] = get_MicroRNA(cancer_type=cancer_type, endpoint=endpoint, 
                                            groups=groups, genders=genders,
                                            FeatureMethod=FeatureMethod, FeatureTrain=False)
                elif Feature == 'Protein':
                    datasets['Protein'] = get_protein(cancer_type=cancer_type, endpoint=endpoint, 
                                            groups=groups, genders=genders,
                                            FeatureMethod=FeatureMethod, FeatureTrain=False)
                elif Feature == 'Methylation':
                    datasets['Methylation'] = get_Methylation(cancer_type=cancer_type, endpoint=endpoint, 
                                            groups=groups, genders=genders,
                                            FeatureMethod=FeatureMethod, FeatureTrain=False)
        # Standardize dataset
        for key in datasets:
            datasets[key] = standarize_dataset(datasets[key])
        
        if FeatureMethod==0:
            print('p-Value based Feature selection will be applied')
            dataset = merge_datasets(datasets)
            k = features_count # feature selection to be done for this number
            
        if FeatureMethod==1:
            print('PCA will be used for Feature extraction')
            
            ######### I will write it later #########
        
        if FeatureMethod == 2:
            print('Autoencoder will be used for Feature extraction')
            
            if AutoencoderSettings==1:
                EncActiv = 'linear'
                DecActiv = 'linear'
                loss_fn = 'mean_squared_error'
            elif AutoencoderSettings==2:
                EncActiv = 'relu'
                DecActiv = 'sigmoid'
                loss_fn = 'binary_crossentropy'
                
            # Load or train encoders
            encoders = load_or_train_encoders(omics_feature, folderISAAC, cancer_type, endpoint, 
                                              groups, genders, data_Category, features_count, 
                                              AE_iter, AE_batchsize, FeatureMethod,
                                              EncActiv, DecActiv, loss_fn, AutoencoderSettings)
            dataset = process_omics(datasets, encoders, omics_feature)
            k = -1 # this depicts that Feature extraction is done or not require now onwards
            
        ## Independent Learning datasets ##
        
        if data_Category in ['Race', 'GenderRace']:
            # Independent - WHITE
            data_WHITE = get_independent_data_single(dataset, data_Category, 'WHITE', groups, genders)
            data_WHITE = get_n_years(data_WHITE, years)
            # Independent - DDP
            data_DDP = get_independent_data_single(dataset, data_Category, 'DDP', groups, genders)
            data_DDP = get_n_years(data_DDP, years)
        
        if data_Category=='GenderRace':
            # Independent - WHITE-FEMALE
            data_WHITE_F = get_independent_data_single(dataset, data_Category, 'WHITE-FEMALE', groups, genders)
            data_WHITE_F = get_n_years(data_WHITE_F, years)
            # Independent - WHITE-MALE
            data_WHITE_M = get_independent_data_single(dataset, data_Category, 'WHITE-MALE', groups, genders)
            data_WHITE_M = get_n_years(data_WHITE_M, years)
            # Independent - DDP-FEMALE
            data_DDP_F = get_independent_data_single(dataset, data_Category, 'DDP-FEMALE', groups, genders)
            data_DDP_F = get_n_years(data_DDP_F, years)
            # Independent - DDP-MALE
            data_DDP_M = get_independent_data_single(dataset, data_Category, 'DDP-MALE', groups, genders)
            data_DDP_M = get_n_years(data_DDP_M, years)
        
        if data_Category=='Gender':
            # Independent - FEMALE
            data_F = get_independent_data_single(dataset, data_Category, 'FEMALE', groups, genders)
            data_F = get_n_years(data_F, years)
            # Independent - MALE
            data_M = get_independent_data_single(dataset, data_Category, 'MALE', groups, genders)
            data_M = get_n_years(data_M, years)
        
        ## Mixture, Naive Transfer, and Transfer Learning dataset ##
        dataset_tl_ccsa = normalize_dataset(dataset)
        dataset_tl_ccsa = get_n_years(dataset_tl_ccsa, years)
        
        dataset = get_n_years(dataset, years)
        
        X, Y, R, y_strat, G, Gy_strat, GRy_strat = dataset
        df = pd.DataFrame(y_strat, columns=['RY'])
        df['GRY'] = GRy_strat
        df['GY'] = Gy_strat
        df['R'] = R
        df['G'] = G
        df['Y'] = Y
        print(X.shape)
        print(df['GRY'].value_counts())#gender with prognosis counts
        print(df['GY'].value_counts())#gender with prognosis counts
        print(df['G'].value_counts())#gender counts
        print(df['RY'].value_counts())#race with prognosis counts
        print(df['R'].value_counts())#race counts
        print(df['Y'].value_counts())#progonsis counts
        
        ###############################
        # parameters #
        ###############################
        parametrs_mix = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,'momentum':0.9, 'learning_rate':0.01,
                        'lr_decay':0.03, 'dropout':0.5, 'L1_reg': 0.001,'L2_reg': 0.001, 'hiddenLayers': [128,64]}
        parameters_MAJ = {'fold':3, 'k':k, 'batch_size':20, 'lr_decay':0.03, 'val_size':0.0, 'learning_rate':0.01,
                        'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_MIN = {'fold':3, 'k':k, 'batch_size':4, 'lr_decay':0.03, 'val_size':0.0, 'learning_rate':0.01,
                        'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_NT  = {'fold':3, 'k':k, 'batch_size':20, 'momentum':0.9, 'lr_decay':0.03, 'val_size':0.0,
                        'learning_rate':0.01, 'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_TL1 = {'fold':3, 'k':k, 'batch_size':20, 'momentum':0.9, 'lr_decay':0.03, 'val_size':0.0,
                        'learning_rate':0.01, 'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64],
                        'train_epoch':100, 'tune_epoch':100, 'tune_lr':0.002, 'tune_batch':10}
        parameters_TL2 = {'fold':3, 'k':k, 'batch_size':10, 'lr_decay':0.03, 'val_size':0.0, 'learning_rate':0.002,
                        'n_epochs':100, 'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_TL3 = {'fold':3, 'n_features':k, 'alpha':0.3, 'batch_size':20, 'learning_rate':0.01, 'hiddenLayers':[100],
                        'dr':0.5, 'momentum':0.9, 'decay':0.03, 'sample_per_class':2, 'SourcePairs':False}
        parameters_TL4 = {'fold':3, 'n_features':k, 'alpha':0.25, 'batch_size':20, 'learning_rate':0.01, 'hiddenLayers':[128,64],
                        'dr':0.5, 'momentum':0.9, 'decay':0.03, 'sample_per_class':2, 'EarlyStop':False,
                        'L1_reg':0.001, 'L2_reg':0.001, 'patience':100, 'n_epochs':100}
        
        res = pd.DataFrame()
        
        for i in range(20):
            
            print('###########################')
            print('Interation no.: '+str(i+1))
            print('###########################')
            
            seed = i
            start_iter = time.time()
            
            df_mix = run_mixture_cv(seed, dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parametrs_mix)
            print('Mixture is done')
            
            if data_Category in ["Race","GenderRace"]:
                
                df_w = run_one_race_cv(seed, data_WHITE, omics_feature, FeatureMethod, **parameters_MAJ)
                df_w = df_w.rename(columns={"Auc": "W_ind"})
                print('Independent WHITE is done.')
                
                df_b = run_one_race_cv(seed, data_DDP, omics_feature, FeatureMethod, **parameters_MIN)
                df_b = df_b.rename(columns={"Auc": "B_ind"})
                print('Independent DDP is done.')
                
                df_nt = run_naive_transfer_cv(seed, 'WHITE', 'DDP', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_NT)
                df_nt = df_nt.rename(columns={"NT_Auc": "NT_Auc"})
                print('Naive Transfer WHITE-DDP is done.')
                
                df_tl_sup = run_supervised_transfer_cv(seed, 'WHITE', 'DDP', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL1)
                df_tl_sup = df_tl_sup.rename(columns={"TL_Auc": "TL_sup"})
                print('Supervised Transfer WHITE-DDP is done.')
                
                df_tl_unsup = run_unsupervised_transfer_cv(seed, 'WHITE', 'DDP', dataset_tl_ccsa, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL2)
                df_tl_unsup = df_tl_unsup.rename(columns={"TL_Auc": "TL_unsup"})
                print('Unsupervised Transfer WHITE-DDP is done.')
                
                df_tl_ccsa = run_CCSA_transfer(seed, 'WHITE', 'DDP', dataset_tl_ccsa, groups, genders, CCSA_path, data_Category, omics_feature, FeatureMethod, **parameters_TL3)
                df_tl_ccsa = df_tl_ccsa.rename(columns={"TL_Auc": "TL_ccsa"})
                print('CCSA Transfer WHITE-DDP is done.')
                
                df_tl_fada = FADA_classification(seed, 'WHITE', 'DDP', dataset, groups, genders, checkpt_path, data_Category, omics_feature, FeatureMethod, **parameters_TL4)
                df_tl_fada = df_tl_fada.rename(columns={"TL_DCD_Auc":"TL_FADA"})
                print('FADA Transfer WHITE-DDP is done.')
                
            if data_Category=='GenderRace':
                
                df_wf = run_one_race_cv(seed, data_WHITE_F, omics_feature, FeatureMethod, **parameters_MAJ)
                df_wf = df_wf.rename(columns={"Auc": "WF_ind"})
                print('Independent WHITE(F) is done.')
                
                df_bf = run_one_race_cv(seed, data_DDP_F, omics_feature, FeatureMethod, **parameters_MIN)
                df_bf = df_bf.rename(columns={"Auc": "BF_ind"})
                print('Independent DDP(F) is done.')
                
                df_wm = run_one_race_cv(seed, data_WHITE_M, omics_feature, FeatureMethod, **parameters_MAJ)
                df_wm = df_wm.rename(columns={"Auc": "WM_ind"})
                print('Independent WHITE(M) is done.')
                
                df_bm = run_one_race_cv(seed, data_DDP_M, omics_feature, FeatureMethod, **parameters_MIN)
                df_bm = df_bm.rename(columns={"Auc": "BM_ind"})
                print('Independent DDP(M) is done.')
                
                df_nt_f = run_naive_transfer_cv(seed, 'WHITE', 'DDP-FEMALE', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_NT)
                df_nt_f = df_nt_f.rename(columns={"NT_Auc": "NT_Auc_DDP(F)"})
                print('Naive Transfer is done for WHITE--DDP(F).')
                
                df_nt_m = run_naive_transfer_cv(seed, 'WHITE', 'DDP-MALE', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_NT)
                df_nt_m = df_nt_m.rename(columns={"NT_Auc": "NT_Auc_DDP(M)"})
                print('Naive Transfer is done for WHITE--DDP(M).')
                
                df_tl_sup_WHITE_DDP_F = run_supervised_transfer_cv(seed, 'WHITE', 'DDP-FEMALE', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL1)
                df_tl_sup_WHITE_DDP_F = df_tl_sup_WHITE_DDP_F.rename(columns={"TL_Auc": "TL_sup_WHITE_DDP(F)"})
                print('Supervised is done for WHITE--DDP(F).')
                
                df_tl_sup_WHITE_DDP_M = run_supervised_transfer_cv(seed, 'WHITE', 'DDP-MALE', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL1)
                df_tl_sup_WHITE_DDP_M = df_tl_sup_WHITE_DDP_M.rename(columns={"TL_Auc": "TL_sup_WHITE_DDP(M)"})
                print('Supervised is done for WHITE--DDP(M).')
                
                df_tl_unsup_WHITE_DDP_F = run_unsupervised_transfer_cv(seed, 'WHITE', 'DDP-FEMALE', dataset_tl_ccsa, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL2)
                df_tl_unsup_WHITE_DDP_F = df_tl_unsup_WHITE_DDP_F.rename(columns={"TL_Auc": "TL_unsup_WHITE_DDP(F)"})
                print('Unsupervised is done for WHITE--DDP(F).')
                
                df_tl_unsup_WHITE_DDP_M = run_unsupervised_transfer_cv(seed, 'WHITE', 'DDP-MALE', dataset_tl_ccsa, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL2)
                df_tl_unsup_WHITE_DDP_M = df_tl_unsup_WHITE_DDP_M.rename(columns={"TL_Auc": "TL_unsup_WHITE_DDP(M)"})
                print('Unsupervised is done for WHITE--DDP(M).')
                
                df_tl_ccsa_WHITE_DDP_F = run_CCSA_transfer(seed, 'WHITE', 'DDP-FEMALE', dataset_tl_ccsa, groups, genders, CCSA_path, data_Category, omics_feature, FeatureMethod, **parameters_TL3)
                df_tl_ccsa_WHITE_DDP_F = df_tl_ccsa_WHITE_DDP_F.rename(columns={"TL_Auc": "TL_ccsa_WHITE_DDP(F)"})
                print('CCSA is done for WHITE--DDP(F).')
                
                df_tl_ccsa_WHITE_DDP_M = run_CCSA_transfer(seed, 'WHITE', 'DDP-MALE', dataset_tl_ccsa, groups, genders, CCSA_path, data_Category, omics_feature, FeatureMethod, **parameters_TL3)
                df_tl_ccsa_WHITE_DDP_M = df_tl_ccsa_WHITE_DDP_M.rename(columns={"TL_Auc": "TL_ccsa_WHITE_DDP(M)"})
                print('CCSA is done for WHITE--DDP(M).')
                
                df_tl_fada_WHITE_DDP_F = FADA_classification(seed, 'WHITE', 'DDP-FEMALE', dataset, groups, genders, checkpt_path, data_Category, omics_feature, FeatureMethod, **parameters_TL4)
                df_tl_fada_WHITE_DDP_F = df_tl_fada_WHITE_DDP_F.rename(columns={"TL_DCD_Auc": "TL_FADA_WHITE_DDP(F)"})
                print('FADA is done for WHITE--DDP(F).')
                
                df_tl_fada_WHITE_DDP_M = FADA_classification(seed, 'WHITE', 'DDP-MALE', dataset, groups, genders, checkpt_path, data_Category, omics_feature, FeatureMethod, **parameters_TL4)
                df_tl_fada_WHITE_DDP_M = df_tl_fada_WHITE_DDP_M.rename(columns={"TL_DCD_Auc": "TL_FADA_WHITE_DDP(M)"})
                print('FADA is done for WHITE--DDP(M).')
                
            if data_Category=='Gender':
                
                df_f = run_one_race_cv(seed, data_F, omics_feature, FeatureMethod, **parameters_MAJ)
                df_f = df_f.rename(columns={"Auc": "F_ind"})
                print('Independent FEMALE is done.')
                
                df_m = run_one_race_cv(seed, data_M, omics_feature, FeatureMethod, **parameters_MIN)
                df_m = df_m.rename(columns={"Auc": "M_ind"})
                print('Independent MALE is done.')
                
                df_nt_F_to_M = run_naive_transfer_cv(seed, 'FEMALE', 'MALE', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_NT)
                df_nt_F_to_M = df_nt_F_to_M.rename(columns={"NT_Auc": "NT_Auc_F_to_M"})
                print('Naive Transfer is done for FEMALE--MALE.')
                
                df_nt_M_to_F = run_naive_transfer_cv(seed, 'MALE', 'FEMALE', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_NT)
                df_nt_M_to_F = df_nt_M_to_F.rename(columns={"NT_Auc": "NT_Auc_M_to_F"})
                print('Naive Transfer is done for MALE--FEMALE.')
                
                df_tl_sup_F_to_M = run_supervised_transfer_cv(seed, 'FEMALE', 'MALE', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL1)
                df_tl_sup_F_to_M = df_tl_sup_F_to_M.rename(columns={"TL_Auc": "TL_sup_F_to_M"})
                print('Supervised is done for FEMALE--MALE.')
                
                df_tl_sup_M_to_F = run_supervised_transfer_cv(seed, 'MALE', 'FEMALE', dataset, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL1)
                df_tl_sup_M_to_F = df_tl_sup_M_to_F.rename(columns={"TL_Auc": "TL_sup_M_to_F"})
                print('Supervised is done for MALE--FEMALE.')
                
                df_tl_unsup_F_to_M = run_unsupervised_transfer_cv(seed, 'FEMALE', 'MALE', dataset_tl_ccsa, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL2)
                df_tl_unsup_F_to_M = df_tl_unsup_F_to_M.rename(columns={"TL_Auc": "TL_unsup_F_to_M"})
                print('Unsupervised is done for FEMALE--MALE.')
                
                df_tl_unsup_M_to_F = run_unsupervised_transfer_cv(seed, 'MALE', 'FEMALE', dataset_tl_ccsa, groups, genders, data_Category, omics_feature, FeatureMethod, **parameters_TL2)
                df_tl_unsup_M_to_F = df_tl_unsup_M_to_F.rename(columns={"TL_Auc": "TL_unsup_M_to_F"})
                print('Unsupervised is done for MALE--FEMALE.')
                
                df_tl_ccsa_F_to_M = run_CCSA_transfer(seed, 'FEMALE', 'MALE', dataset_tl_ccsa, groups, genders, CCSA_path, data_Category, omics_feature, FeatureMethod, **parameters_TL3)
                df_tl_ccsa_F_to_M = df_tl_ccsa_F_to_M.rename(columns={"TL_Auc": "TL_ccsa_F_to_M"})
                print('CCSA is done for FEMALE--MALE.')
                
                df_tl_ccsa_M_to_F = run_CCSA_transfer(seed, 'MALE', 'FEMALE', dataset_tl_ccsa, groups, genders, CCSA_path, data_Category, omics_feature, FeatureMethod, **parameters_TL3)
                df_tl_ccsa_M_to_F = df_tl_ccsa_M_to_F.rename(columns={"TL_Auc": "TL_ccsa_M_to_F"})
                print('CCSA is done for MALE--FEMALE.')
                
                df_tl_fada_F_to_M = FADA_classification(seed, 'FEMALE', 'MALE', dataset, groups, genders, checkpt_path, data_Category, omics_feature, FeatureMethod, **parameters_TL4)
                df_tl_fada_F_to_M = df_tl_fada_F_to_M.rename(columns={"TL_DCD_Auc": "TL_FADA_F_to_M"})
                print('FADA is done for FEMALE--MALE.')
                
                df_tl_fada_M_to_F = FADA_classification(seed, 'MALE', 'FEMALE', dataset, groups, genders, checkpt_path, data_Category, omics_feature, FeatureMethod, **parameters_TL4)
                df_tl_fada_M_to_F = df_tl_fada_M_to_F.rename(columns={"TL_DCD_Auc": "TL_FADA_M_to_F"})
                print('FADA is done for MALE--FEMALE.')
                
            end_iter = time.time()
            print("The time of loop execution is :", end_iter-start_iter)
            timeFor_iter = pd.DataFrame({'Time':[end_iter - start_iter]},index=[seed])
            
            if data_Category == 'Race':
                df1 = pd.concat([timeFor_iter,
                                 df_mix,
                                 df_w, df_b,
                                 df_nt,
                                 df_tl_sup,
                                 df_tl_unsup,
                                 df_tl_ccsa,
                                 df_tl_fada
                                 ], sort=False, axis=1)
            
            elif data_Category == 'GenderRace':
                df1 = pd.concat([timeFor_iter,
                                 df_mix,
                                 df_w, df_b, df_wf, df_bf, df_wm, df_bm,
                                 df_nt, df_nt_f, df_nt_m,
                                 df_tl_sup, df_tl_sup_WHITE_DDP_F, df_tl_sup_WHITE_DDP_M,
                                 df_tl_unsup, df_tl_unsup_WHITE_DDP_F, df_tl_unsup_WHITE_DDP_M,
                                 df_tl_ccsa, df_tl_ccsa_WHITE_DDP_F, df_tl_ccsa_WHITE_DDP_M,
                                 df_tl_fada, df_tl_fada_WHITE_DDP_F, df_tl_fada_WHITE_DDP_M
                                 ], sort=False, axis=1)
            
            elif data_Category == 'Gender':
                df1 = pd.concat([timeFor_iter,
                                 df_mix,
                                 df_f, df_m,
                                 df_nt_M_to_F, df_nt_F_to_M,
                                 df_tl_sup_F_to_M, df_tl_sup_M_to_F,
                                 df_tl_unsup_F_to_M, df_tl_unsup_M_to_F,
                                 df_tl_ccsa_F_to_M, df_tl_ccsa_M_to_F,
                                 df_tl_fada_F_to_M, df_tl_fada_M_to_F
                                 ], sort=False, axis=1)
            
            print(df1)
            res = res.append(df1)
        
        res.to_excel(out_file_name)

if __name__ == '__main__':
    main()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            