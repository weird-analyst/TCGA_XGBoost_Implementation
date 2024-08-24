# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:56:30 2024

@author: teesh
"""

import theano
from keras import Input, Model
from keras.layers import Dropout, Dense, Activation, Lambda
from keras.optimizers import SGD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
from model.CCSA import Initialization
from model.mlp import get_k_best, MLP
# from sklearn.decomposition import PCA
import torch
import dataloader
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from model import fada_model
use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)

feature_counts = {
    'Protein': 189,
    'mRNA': 17176,
    'MicroRNA': 662,
    'Methylation': 11882
}

def run_cv(seed, fold, X, Y, R, y_strat, G, Gy_strat, GRy_strat,
           omics_feature, FeatureMethod,
           val_size=0, pretrain_set=None,
           batch_size=32, k=-1,
           learning_rate=0.01, lr_decay=0.0,
           dropout=0.5, n_epochs=100, momentum=0.9,
           L1_reg=0.001, L2_reg=0.001,
           hiddenLayers=[128, 64]):
    
    # Pretrain set processing
    X_w = pretrain_set.get_value(borrow=True) if pretrain_set else None

    # Initialize DataFrame for results
    if len(np.shape(omics_feature)) == 0:  # single omics
        m = X.shape[1] if k < 0 else min(X.shape[1], k)
        columns = list(range(m))
    else:  # multi omics
        m_list = []
        for feature in omics_feature:
            f_num = feature_counts[feature]
            m_list.append(min(f_num, k) if k > 0 else f_num)
        m = sum(m_list)
        columns = list(range(m))
    
    columns.extend(['scr', 'R', 'G', 'Y'])
    df = pd.DataFrame(columns=columns)
    
    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    
    for train_index, test_index in kf.split(X, GRy_strat):
        # Split data into training and test sets
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        R_train, R_test = R[train_index], R[test_index]
        y_strat_train, y_strat_test = y_strat[train_index], y_strat[test_index]
        G_train, G_test = G[train_index], G[test_index]
        Gy_strat_train, Gy_strat_test = Gy_strat[train_index], Gy_strat[test_index]
        GRy_strat_train, GRy_strat_test = GRy_strat[train_index], GRy_strat[test_index]
        
        if FeatureMethod == 0:  # ANOVA based feature selection
            if k > 0:  # k = features_count from the main function provided by user
                if len(np.shape(omics_feature)) == 0:  # single omics
                    if X_train.shape[1] > k:  # Check if the number of features in the training set is greater than k
                        print(f"Performing ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train.shape[1]}) in the training set is more than k ({k}).")
                        k_best = SelectKBest(f_classif, k=k)
                        k_best.fit(X_train, Y_train)
                        X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)
                    else:
                        print(f"Skipping ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train.shape[1]}) in the training set is less than or equal to k ({k}).")
                        # Skip feature selection and use all features
                        X_train, X_test = X_train, X_test
                else:  # multi omics
                    k_best_list = []
                    start_idx = 0
                    for feature in omics_feature:
                        f_num = feature_counts[feature]
                        end_idx = start_idx + f_num
                        actual_feature_count = X_train[:, start_idx:end_idx].shape[1]
                        if k < f_num:  # Check if k is less than the number of features in the dataset
                            print(f"Performing ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is more than k ({k}).")
                            k_best = SelectKBest(f_classif, k=k)
                            k_best.fit(X_train[:, start_idx:end_idx], Y_train)
                            k_best_list.append(k_best)
                        else:  # No feature selection needed
                            print(f"Skipping ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is less than or equal to k ({k}).")
                            k_best_list.append(None)
                        start_idx = end_idx
                    
                    # Transform and concatenate the features
                    X_train_transformed = []
                    X_test_transformed = []
                    start_idx = 0
                    for i, feature in enumerate(omics_feature):
                        f_num = feature_counts[feature]
                        end_idx = start_idx + f_num
                        if k_best_list[i] is not None:  # If feature selection was applied
                            X_train_transformed.append(k_best_list[i].transform(X_train[:, start_idx:end_idx]))
                            X_test_transformed.append(k_best_list[i].transform(X_test[:, start_idx:end_idx]))
                        else:  # No feature selection, use original data
                            X_train_transformed.append(X_train[:, start_idx:end_idx])
                            X_test_transformed.append(X_test[:, start_idx:end_idx])
                        start_idx = end_idx
                    
                    X_train = np.concatenate(X_train_transformed, axis=1)
                    X_test = np.concatenate(X_test_transformed, axis=1)
            
            if pretrain_set:
                if len(np.shape(omics_feature)) == 0:  # single omics
                    if k_best is not None:
                        X_base = k_best.transform(X_w)
                    else:
                        X_base = X_w
                else:  # multi omics
                    X_w_transformed = []
                    start_idx = 0
                    for i, feature in enumerate(omics_feature):
                        f_num = feature_counts[feature]
                        end_idx = start_idx + f_num
                        if k_best_list[i] is not None:  # If feature selection was applied
                            X_w_transformed.append(k_best_list[i].transform(X_w[:, start_idx:end_idx]))
                        else:  # No feature selection, use original data
                            X_w_transformed.append(X_w[:, start_idx:end_idx])
                        start_idx = end_idx
                    X_base = np.concatenate(X_w_transformed, axis=1)
                pretrain_set = theano.shared(np.array(X_base), name='pretrain_set', borrow=True)
        
        # Split validation data if val_size is specified
        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val, R_train, R_val, G_train, G_val, \
                y_strat_train, y_strat_val, Gy_strat_train, Gy_strat_val, \
                GRy_strat_train, GRy_strat_val = \
                train_test_split(X_train, Y_train, R_train, y_strat_train, \
                                 G_train, Gy_strat_train, GRy_strat_train)
            valid_data = (X_val, Y_val)
        
        train_data = (X_train, Y_train)
        n_in = X_train.shape[1]
        
        # Initialize the classifier
        classifier = MLP(n_in=n_in, learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout, 
                         L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers, momentum=momentum)
        
        if pretrain_set:
            pretrain_config = {'pt_batchsize': 32, 'pt_lr': 0.01, 'pt_epochs': 500, 'corruption_level': 0.3}
            classifier.pretrain(pretrain_set=pretrain_set, pretrain_config=pretrain_config)
            classifier.tune(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        else:
            classifier.train(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        
        # Get scores and construct the result DataFrame
        X_scr = classifier.get_score(X_test)
        array1 = np.column_stack((X_test, X_scr[:,1], R_test, G_test, Y_test))
        df_temp1 = pd.DataFrame(array1, index=list(test_index), columns=columns)
        df = df.append(df_temp1)
        
    return df

def run_mixture_cv(seed, dataset, groups, genders,
                   data_Category, omics_feature, FeatureMethod, 
                   fold=3, k=-1,
                   val_size=0, batch_size=32, momentum=0.9,
                   learning_rate=0.01, lr_decay=0.0,
                   dropout=0.5, n_epochs=100, save_to=None,
                   L1_reg=0.001, L2_reg=0.001, 
                   hiddenLayers=[128, 64]):
    
    X, Y, R, y_strat, G, Gy_strat, GRy_strat = dataset
    df = run_cv(seed, fold, X, Y, R, y_strat, G, Gy_strat, GRy_strat,
                omics_feature, FeatureMethod,
                val_size=val_size, batch_size=batch_size, k=k, momentum=momentum,
                learning_rate=learning_rate, lr_decay=lr_decay,
                dropout=dropout, n_epochs=n_epochs,
                L1_reg=L1_reg, L2_reg=L2_reg, hiddenLayers=hiddenLayers)
    
    if save_to:
        df.to_csv(save_to)
    
    # Mixture 0 - AUROC for EA+DDP
    y_test_0, y_scr_0 = list(df['Y'].values), list(df['scr'].values)
    
    res = {}
    
    if data_Category in ['Race', 'GenderRace']:
        # Mixture 1 - AUROC for EA
        y_test_1 = list(df.loc[df['R'].isin(groups[0]), 'Y'].values)
        y_scr_1 = list(df.loc[df['R'].isin(groups[0]), 'scr'].values)
        # Mixture 2 - AUROC for DDP
        y_test_2 = list(df.loc[df['R'].isin(groups[1]), 'Y'].values)
        y_scr_2 = list(df.loc[df['R'].isin(groups[1]), 'scr'].values)
        
        res.update({
            'A_Auc': roc_auc_score(y_test_0, y_scr_0, average='weighted'),
            'W_Auc': roc_auc_score(y_test_1, y_scr_1, average='weighted'),
            'B_Auc': roc_auc_score(y_test_2, y_scr_2, average='weighted')
        })
    
    if data_Category == 'GenderRace':
        df['RG'] = df['R'].astype(str) + df['G'].astype(str)
        temp1 = groups[0] + genders[1] # 'WHITE-FEMALE'
        temp2 = groups[0] + genders[0] # 'WHITE-MALE'
        temp3 = groups[1] + genders[0] # 'DDP-MALE'
        temp4 = groups[1] + genders[1] # 'DDP-FEMALE'
        # AUROC for EA(F)
        y_test_3 = list(df.loc[df['RG'] == temp1, 'Y'].values)
        y_scr_3 = list(df.loc[df['RG'] == temp1, 'scr'].values)
        # AUROC for DDP(F)
        y_test_4 = list(df.loc[df['RG'] == temp4, 'Y'].values)
        y_scr_4 = list(df.loc[df['RG'] == temp4, 'scr'].values)
        # AUROC for EA(M)
        y_test_5 = list(df.loc[df['RG'] == temp2, 'Y'].values)
        y_scr_5 = list(df.loc[df['RG'] == temp2, 'scr'].values)
        # AUROC for DDP(M)
        y_test_6 = list(df.loc[df['RG'] == temp3, 'Y'].values)
        y_scr_6 = list(df.loc[df['RG'] == temp3, 'scr'].values)
        
        res.update({
            'A_Auc': roc_auc_score(y_test_0, y_scr_0, average='weighted'),
            'WF_Auc': roc_auc_score(y_test_3, y_scr_3, average='weighted'),
            'BF_Auc': roc_auc_score(y_test_4, y_scr_4, average='weighted'),
            'WM_Auc': roc_auc_score(y_test_5, y_scr_5, average='weighted'),
            'BM_Auc': roc_auc_score(y_test_6, y_scr_6, average='weighted')
        })
    
    if data_Category == 'Gender':
        # AUROC for MALE
        y_test_3 = list(df.loc[df['G'].isin([genders[0]]), 'Y'].values)
        y_scr_3 = list(df.loc[df['G'].isin([genders[0]]), 'scr'].values)
        # AUROC for FEMALE
        y_test_4 = list(df.loc[df['G'].isin([genders[1]]), 'Y'].values)
        y_scr_4 = list(df.loc[df['G'].isin([genders[1]]), 'scr'].values)
        
        res.update({
            'A_Auc': roc_auc_score(y_test_0, y_scr_0, average='weighted'),
            'M_Auc': roc_auc_score(y_test_3, y_scr_3, average='weighted'),
            'F_Auc': roc_auc_score(y_test_4, y_scr_4, average='weighted')
        })
        
    df = pd.DataFrame(res, index=[seed])
    
    return df

def run_one_race_cv(seed, dataset, omics_feature, FeatureMethod,
                    fold=3,  k=-1, val_size=0, batch_size=32,
                    learning_rate=0.01, lr_decay=0.0, dropout=0.5, save_to=None,
                    L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):
    
    X, Y, R, y_strat, G, Gy_strat, GRy_strat = dataset
    df = run_cv(seed, fold, X, Y, R, y_strat, G, Gy_strat, GRy_strat,
                omics_feature, FeatureMethod,
                val_size=val_size, batch_size=batch_size, k=k,
                learning_rate=learning_rate,
                lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg,
                hiddenLayers=hiddenLayers)
    
    if save_to:
        df.to_csv(save_to)
    
    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'Auc': A_CI}
    df = pd.DataFrame(res, index=[seed])
    
    return df

def run_CCSA_transfer(seed, Source, Target, dataset, groups, genders,
                      CCSA_path, data_Category, omics_feature, FeatureMethod, n_features,
                      fold=3, alpha=0.25, learning_rate = 0.01,
                      hiddenLayers=[128, 64], dr=0.5,
                      momentum=0.0, decay=0, batch_size=32,
                      sample_per_class=2, repetition=1,
                      SourcePairs=False):
    
    X, Y, R, y_strat, G, Gy_strat, GRy_strat = dataset
    
    df = pd.DataFrame(X)
    df['R'] = R
    df['Y'] = Y
    df['G'] = G
    df['GRY'] = GRy_strat
    
    if data_Category == "GenderRace":
        temp1 = genders[1] + '0' + groups[0] # 'FEMALE0WHITE'
        temp2 = genders[1] + '1' + groups[0] # 'FEMALE1WHITE'
        temp3 = genders[0] + '0' + groups[0] # 'MALE0WHITE'
        temp4 = genders[0] + '1' + groups[0] # 'MALE1WHITE'
        temp5 = genders[1] + '0' + groups[1] # 'FEMALE0DDP'
        temp6 = genders[1] + '1' + groups[1] # 'FEMALE1DDP'
        temp7 = genders[0] + '0' + groups[1] # 'MALE0DDP'
        temp8 = genders[0] + '1' + groups[1] # 'MALE1DDP'
    
    # domain adaptation - source groups
    if Source=='WHITE':
        df_train = df[df['R']==groups[0]]
    elif Source=='DDP':
        df_train = df[df['R']==groups[1]]
    elif Source=='FEMALE':
        df_train = df[df['G']==genders[1]]
    elif Source=='MALE':
        df_train = df[df['G']==genders[0]]
    elif Source=='WHITE-FEMALE':
        df_train = df[df['GRY'].isin([temp1,temp2])]
    elif Source=='WHITE-MALE':
        df_train = df[df['GRY'].isin([temp3,temp4])]
    elif Source=='DDP-FEMALE':
        df_train = df[df['GRY'].isin([temp5,temp6])]
    elif Source=='DDP-MALE':
        df_train = df[df['GRY'].isin([temp7,temp8])]
    
    df_w_y = df_train['Y']
    df_train = df_train.drop(columns=['Y', 'R', 'G', 'GRY'])
    Y_train_source = df_w_y.values.ravel()
    X_train_source = df_train.values
    
 	#test groups
    if Target=='DDP':
        df_test = df[df['R']==groups[1]]
    elif Target=='DDP-FEMALE':
        df_test = df[df['GRY'].isin([temp5,temp6])]
    elif Target=='DDP-MALE':
        df_test = df[df['GRY'].isin([temp7,temp8])]
    elif Target=='FEMALE':
        df_test = df[df['G']==genders[1]]
    elif Target=='MALE':
        df_test = df[df['G']==genders[0]]
    
    df_b_y = df_test['Y']
    df_b_R = df_test['R'].values.ravel()
    df_b_G = df_test['G'].values.ravel()
    df_test = df_test.drop(columns=['Y', 'R', 'G', 'GRY'])
    Y_test = df_b_y.values.ravel()
    X_test = df_test.values
    
    if FeatureMethod == 0:  # ANOVA based feature selection
        if n_features > 0:  # n_features = features_count from the main function provided by user
            if len(np.shape(omics_feature)) == 0:  # single omics
                if X_train_source.shape[1] > n_features:  # Check if the number of features in the training set is greater than n_features
                    print(f"Performing ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train_source.shape[1]}) in the training set is more than n_features ({n_features}).")
                    X_train_source, X_test = get_k_best(X_train_source, Y_train_source, X_test, n_features)
                else:
                    print(f"Skipping ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train_source.shape[1]}) in the training set is less than or equal to n_features ({n_features}).")
                    # Skip feature selection and use all features
                    X_train_source, X_test = X_train_source, X_test
            else:  # multi omics
                k_best_list = []
                start_idx = 0
                for feature in omics_feature:
                    f_num = feature_counts[feature]
                    end_idx = start_idx + f_num
                    actual_feature_count = X_train_source[:, start_idx:end_idx].shape[1]
                    if n_features < f_num:  # Check if n_features is less than the number of features in the dataset
                        print(f"Performing ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is more than n_features ({n_features}).")
                        k_best = SelectKBest(f_classif, k=n_features)
                        k_best.fit(X_train_source[:, start_idx:end_idx], Y_train_source)
                        k_best_list.append(k_best)
                    else:  # No feature selection needed
                        print(f"Skipping ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is less than or equal to n_features ({n_features}).")
                        k_best_list.append(None)
                    start_idx = end_idx
                    
                # Transform and concatenate the features
                X_train_source_transformed = []
                X_test_transformed = []
                start_idx = 0
                for i, feature in enumerate(omics_feature):
                    f_num = feature_counts[feature]
                    end_idx = start_idx + f_num
                    if k_best_list[i] is not None:
                        X_train_source_transformed.append(k_best_list[i].transform(X_train_source[:, start_idx:end_idx]))
                        X_test_transformed.append(k_best_list[i].transform(X_test[:, start_idx:end_idx]))
                    else:
                        X_train_source_transformed.append(X_train_source[:, start_idx:end_idx])
                        X_test_transformed.append(X_test[:, start_idx:end_idx])
                    start_idx = end_idx
                X_train_source = np.concatenate(X_train_source_transformed, axis=1)
                X_test = np.concatenate(X_test_transformed, axis=1)
        else:
            print(f"Skipping feature selection as n_features ({n_features}) is not less than the number of features in X_test ({X_test.shape[1]}).")
            n_features = X_test.shape[1]
    
    if sample_per_class==None:
        samples_provided = 'No'
    else: 
        samples_provided = 'Yes'
    
    df_score = pd.DataFrame(columns=['scr', 'Y', 'R', 'G'])
    kf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(X_test, Y_test):
        X_train_target_full, X_test_target = X_test[train_index], X_test[test_index]
        Y_train_target_full, Y_test_target = Y_test[train_index], Y_test[test_index]
        R_train_target_full, R_test_target = df_b_R[train_index], df_b_R[test_index]
        G_train_target_full, G_test_target = df_b_G[train_index], df_b_G[test_index]
        if samples_provided=='No':
            maxallowedsamples_un = np.unique(Y_train_target_full,return_counts=True)
            maxallowedsamples = min(maxallowedsamples_un[1])
            # print('Max samples are allowed: '+ str(maxallowedsamples))
            X_train1,X_val1,Y_train1,Y_val1 = train_test_split(X_train_target_full,Y_train_target_full,random_state=None)
            target_samples_count = np.unique(Y_train1,return_counts=True)
            min_target_samples_count = min(target_samples_count[1])
            sample_per_class = min_target_samples_count
            # print('Sample per class is : '+str(sample_per_class))
            if sample_per_class==1:
                sample_per_class = 2
            elif sample_per_class>2:
                if sample_per_class>maxallowedsamples:
                    sample_per_class = maxallowedsamples
        # print('==================================')
        # print('Sample per class is : '+str(sample_per_class))
        # print('==================================')
        
        index0 = np.where(Y_train_target_full == 0)
        index1 = np.where(Y_train_target_full == 1)

        target_samples = []
        target_samples.extend(index0[0][0:sample_per_class])
        target_samples.extend(index1[0][0:sample_per_class])

        X_train_target = X_train_target_full[target_samples]
        Y_train_target = Y_train_target_full[target_samples]

        X_val_target = [e for idx, e in enumerate(X_train_target_full) if idx not in target_samples]
        Y_val_target = [e for idx, e in enumerate(Y_train_target_full) if idx not in target_samples]
        
        print(np.shape(X_train_target))
        print(np.shape(X_train_source))
        print(np.shape(X_val_target))
        print(np.shape(X_test_target))
        print(n_features)
        
        best_score, best_Auc = train_and_predict(Source, Target, 
                                     X_train_target, Y_train_target,
                                     X_train_source, Y_train_source,
                                     X_val_target, Y_val_target,
                                     X_test_target, Y_test_target,
                                     CCSA_path,
                                     sample_per_class=sample_per_class,
                                     alpha=alpha, learning_rate=learning_rate,
                                     hiddenLayers=hiddenLayers, dr=dr,
                                     momentum=momentum, decay=decay,
                                     batch_size=batch_size,
                                     repetition=repetition,
                                     n_features=np.shape(X_train_target)[1],
                                     SourcePairs=SourcePairs)

        #print (best_score.shape)
        #print (Y_test_target.shape)

        array = np.column_stack((best_score, Y_test_target, R_test_target, G_test_target))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'Y', 'R', 'G'])
        df_score = df_score.append(df_temp)
    
    auc = roc_auc_score(list(df_score['Y'].values), list(df_score['scr'].values))
    res = {'TL_Auc': auc}
    
    df = pd.DataFrame(res, index=[seed])
    #print(res)
    
    return df

def train_and_predict(Source, Target,
                      X_train_target, y_train_target,
                      X_train_source, y_train_source,
                      X_val_target, Y_val_target,
                      X_test, y_test, CCSA_path,
                      repetition, sample_per_class,
                      alpha=0.25, learning_rate = 0.01,
                      hiddenLayers=[100, 50], dr=0.5,
                      momentum=0.0, decay=0, batch_size=32,
                      n_features = 400,
                      SourcePairs=False):
    
    # size of input variable for each patient
    domain_adaptation_task = Source + '_to_' + Target
    input_shape = (n_features,)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # number of classes for digits classification
    nb_classes = 2
    # Loss = (1-alpha)Classification_Loss + (alpha)CSA
    alpha = alpha

    # Having two streams. One for source and one for target.
    model1 = Initialization.Create_Model(hiddenLayers=hiddenLayers, dr=dr)
    processed_a = model1(input_a)
    processed_b = model1(input_b)

    # Creating the prediction function. This corresponds to h in the paper.
    processed_a = Dropout(0.5)(processed_a)
    out1 = Dense(nb_classes)(processed_a)
    out1 = Activation('softmax', name='classification')(out1)

    distance = Lambda(Initialization.euclidean_distance,
                      output_shape=Initialization.eucl_dist_output_shape,
                      name='CSA')([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
    optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay) # momentum=0., decay=0.
    model.compile(loss={'classification': 'binary_crossentropy', 'CSA': Initialization.contrastive_loss},
                  optimizer=optimizer,
                  loss_weights={'classification': 1 - alpha, 'CSA': alpha})

    print('Domain Adaptation Task: ' + domain_adaptation_task)
    # for repetition in range(10):
    Initialization.Create_Pairs(CCSA_path, domain_adaptation_task,
                                repetition, sample_per_class,
                                X_train_target, y_train_target,
                                X_train_source, y_train_source,
                                n_features=n_features,
                                SourcePairs=SourcePairs)
    best_score, best_Auc = Initialization.training_the_model(model, CCSA_path,
                                                             domain_adaptation_task,
                                                             repetition,
                                                             sample_per_class,batch_size,
                                                             X_val_target,
                                                             Y_val_target,
                                                             X_test, y_test,
                                                             SourcePairs=SourcePairs)

    print('Best AUC for {} target sample per class and repetition {} is {}.'.format(sample_per_class,
                                                                            repetition, best_Auc))
    return best_score, best_Auc

def run_unsupervised_transfer_cv(seed, Source, Target, dataset, groups,
                                 genders, data_Category, omics_feature, 
                                 FeatureMethod, fold=3,
                                 val_size=0, k=-1,
                                 batch_size=32, save_to=None,
                                 learning_rate=0.01, lr_decay=0.0,
                                 dropout=0.5, n_epochs=100,
                                 L1_reg=0.001, L2_reg=0.001,
                                 hiddenLayers=[128, 64]):
    
    X, Y, R, y_strat, G, Gy_strat, GRy_strat = dataset
    
    if data_Category == "GenderRace":
        temp1 = genders[1] + '0' + groups[0] # 'FEMALE0WHITE'
        temp2 = genders[1] + '1' + groups[0] # 'FEMALE1WHITE'
        temp3 = genders[0] + '0' + groups[0] # 'MALE0WHITE'
        temp4 = genders[0] + '1' + groups[0] # 'MALE1WHITE'
        temp5 = genders[1] + '0' + groups[1] # 'FEMALE0DDP'
        temp6 = genders[1] + '1' + groups[1] # 'FEMALE1DDP'
        temp7 = genders[0] + '0' + groups[1] # 'MALE0DDP'
        temp8 = genders[0] + '1' + groups[1] # 'MALE1DDP'
    
    # Determine Source indices
    if Source == 'WHITE-FEMALE':
        idx = ((GRy_strat == temp1) | (GRy_strat == temp2))
    elif Source == 'WHITE-MALE':
        idx = ((GRy_strat == temp3) | (GRy_strat == temp4))
    elif Source == 'DDP-FEMALE':
        idx = ((GRy_strat == temp5) | (GRy_strat == temp6))
    elif Source == 'DDP-MALE':
        idx = ((GRy_strat == temp7) | (GRy_strat == temp8))
    elif Source=='WHITE':
        idx = (R==groups[0])
    elif Source=='DDP':
        idx = (R==groups[1])
    elif Source=='FEMALE':
        idx = (G==genders[1])
    elif Source=='MALE':
        idx = (G==genders[0]) 
    
    X_s, Y_s = X[idx==True], Y[idx==True]
    pretrain_set = (X_s, Y_s)

    if Target=='DDP-FEMALE':
        idx = ((GRy_strat==temp5)|(GRy_strat==temp6))
    elif Target=='DDP-MALE':
        idx = ((GRy_strat==temp7)|(GRy_strat==temp8))
    elif Target=='FEMALE':
        idx = (G==genders[1])
    elif Target=='MALE':
        idx = (G==genders[0])
    elif Target=='DDP':
        idx = (R==groups[1])
    X_t, Y_t, R_t, y_strat_t, G_t, Gy_strat_t, GRy_strat_t = X[idx==True], Y[idx==True], R[idx==True], y_strat[idx==True], G[idx==True], Gy_strat[idx==True], GRy_strat[idx==True]
    
    pretrain_set = theano.shared(X_s, name='pretrain_set', borrow=True)
    #print('pretrain_set in unsupervised loop is:')
    #print(pretrain_set)
    
    df = run_cv(seed, fold, X_t, Y_t, R_t, y_strat_t, G_t, Gy_strat_t, GRy_strat_t,
                omics_feature, FeatureMethod, pretrain_set=pretrain_set,
                val_size=val_size, batch_size=batch_size, k=k, n_epochs=n_epochs,
                learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg, hiddenLayers=hiddenLayers)
    
    if save_to:
        df.to_csv(save_to)
    
    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'TL_Auc': A_CI}
    
    df = pd.DataFrame(res, index=[seed])
    
    return df

def run_supervised_transfer_cv(seed, Source, Target, dataset,
                               groups, genders, data_Category,
                               omics_feature, FeatureMethod, fold=3,
                               val_size=0, k=-1,
                               batch_size=32,
                               learning_rate=0.01, lr_decay=0.0,
                               dropout=0.5, tune_epoch=200,
                               tune_lr=0.002, train_epoch=1000,
                               L1_reg=0.001, L2_reg=0.001,
                               hiddenLayers=[128, 64], tune_batch=10,
                               momentum=0.9):
    
    X, Y, R, y_strat, G, Gy_strat, GRy_strat = dataset
    
    if data_Category == "GenderRace":
        temp1 = genders[1] + '0' + groups[0] # 'FEMALE0WHITE'
        temp2 = genders[1] + '1' + groups[0] # 'FEMALE1WHITE'
        temp3 = genders[0] + '0' + groups[0] # 'MALE0WHITE'
        temp4 = genders[0] + '1' + groups[0] # 'MALE1WHITE'
        temp5 = genders[1] + '0' + groups[1] # 'FEMALE0DDP'
        temp6 = genders[1] + '1' + groups[1] # 'FEMALE1DDP'
        temp7 = genders[0] + '0' + groups[1] # 'MALE0DDP'
        temp8 = genders[0] + '1' + groups[1] # 'MALE1DDP'
    
    # Determine Source indices
    if Source == 'WHITE-FEMALE':
        idx = ((GRy_strat == temp1) | (GRy_strat == temp2))
    elif Source == 'WHITE-MALE':
        idx = ((GRy_strat == temp3) | (GRy_strat == temp4))
    elif Source == 'DDP-FEMALE':
        idx = ((GRy_strat == temp5) | (GRy_strat == temp6))
    elif Source == 'DDP-MALE':
        idx = ((GRy_strat == temp7) | (GRy_strat == temp8))
    elif Source=='WHITE':
        idx = (R==groups[0])
    elif Source=='DDP':
        idx = (R==groups[1])
    elif Source=='FEMALE':
        idx = (G==genders[1])
    elif Source=='MALE':
        idx = (G==genders[0]) 
    
    X_s, Y_s, R_s, y_strat_s, G_s, Gy_strat_s, GRy_strat_s = X[idx==True], Y[idx==True], R[idx==True], y_strat[idx==True], G[idx==True], Gy_strat[idx==True], GRy_strat[idx==True]
    pretrain_set = (X_s, Y_s)
    
    if Target=='DDP-FEMALE':
        idx = ((GRy_strat==temp5)|(GRy_strat==temp6))
    elif Target=='DDP-MALE':
        idx = ((GRy_strat==temp7)|(GRy_strat==temp8))
    elif Target=='FEMALE':
        idx = (G==genders[1])
    elif Target=='MALE':
        idx = (G==genders[0])
    elif Target=='DDP':
        idx = (R==groups[1])
    X_t, Y_t, R_t, y_strat_t, G_t, Gy_strat_t, GRy_strat_t = X[idx==True], Y[idx==True], R[idx==True], y_strat[idx==True], G[idx==True], Gy_strat[idx==True], GRy_strat[idx==True]
    
    df = pd.DataFrame(columns=['scr', 'R', 'G', 'Y'])
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    
    for train_index, test_index in kf.split(X_t, GRy_strat_t):
        X_train, X_test = X_t[train_index], X_t[test_index]
        Y_train, Y_test = Y_t[train_index], Y_t[test_index]
        R_train, R_test = R_t[train_index], R_t[test_index]
        G_train, G_test = G_t[train_index], G_t[test_index]
        y_strat_train, y_strat_test = y_strat_t[train_index], y_strat_t[test_index]
        Gy_strat_train, Gy_strat_test = Gy_strat_t[train_index], Gy_strat_t[test_index]
        GRy_strat_train, GRy_strat_test = GRy_strat_t[train_index], GRy_strat_t[test_index]
        
        if FeatureMethod == 0:  # ANOVA based feature selection
            if k > 0:  # k = features_count from the main function provided by user
                if len(np.shape(omics_feature)) == 0:  # single omics
                    if X_train.shape[1] > k:  # Check if the number of features in the training set is greater than k
                        print(f"Performing ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train.shape[1]}) in the training set is more than k ({k}).")
                        k_best = SelectKBest(f_classif, k=k)
                        k_best.fit(X_train, Y_train)
                        X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)
                        X_base = k_best.transform(X_s)
                    else:
                        print(f"Skipping ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train.shape[1]}) in the training set is less than or equal to k ({k}).")
                        # Use all features
                        X_base = X_s
                else:  # multi omics
                    k_best_list = []
                    start_idx = 0
                    for feature in omics_feature:
                        f_num = feature_counts[feature]
                        end_idx = start_idx + f_num
                        actual_feature_count = X_train[:, start_idx:end_idx].shape[1]
                        if k < actual_feature_count:  # Check if k is less than the number of features in the dataset
                            print(f"Performing ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is more than k ({k}).")
                            k_best = SelectKBest(f_classif, k=k)
                            k_best.fit(X_train[:, start_idx:end_idx], Y_train)
                            k_best_list.append(k_best)
                        else:  # No feature selection needed
                            print(f"Skipping ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is less than or equal to k ({k}).")
                            k_best_list.append(None)
                        start_idx = end_idx
        
                    # Transform and concatenate the features
                    X_train_transformed = []
                    X_test_transformed = []
                    X_s_transformed = []
                    start_idx = 0
                    for i, feature in enumerate(omics_feature):
                        f_num = feature_counts[feature]
                        end_idx = start_idx + f_num
                        if k_best_list[i] is not None:  # If feature selection was applied
                            X_train_transformed.append(k_best_list[i].transform(X_train[:, start_idx:end_idx]))
                            X_test_transformed.append(k_best_list[i].transform(X_test[:, start_idx:end_idx]))
                            X_s_transformed.append(k_best_list[i].transform(X_s[:, start_idx:end_idx]))
                        else:  # No feature selection, use original data
                            X_train_transformed.append(X_train[:, start_idx:end_idx])
                            X_test_transformed.append(X_test[:, start_idx:end_idx])
                            X_s_transformed.append(X_s[:, start_idx:end_idx])
                        start_idx = end_idx
        
                    X_train = np.concatenate(X_train_transformed, axis=1)
                    X_test = np.concatenate(X_test_transformed, axis=1)
                    X_base = np.concatenate(X_s_transformed, axis=1)
        
                pretrain_set = (X_base, Y_s)
        
        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val, R_train, R_val, G_train, G_val, \
                y_strat_train, y_strat_val, Gy_strat_train, Gy_strat_val, \
                GRy_strat_train, GRy_strat_val = \
                train_test_split(X_train, Y_train, R_train, y_strat_train, \
                                 G_train, Gy_strat_train, GRy_strat_train)
            valid_data = (X_val, Y_val)
        
        train_data = (X_train, Y_train)
        n_in = X_train.shape[1]
        classifier = MLP(n_in=n_in, learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout, L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers)
        classifier.train(pretrain_set, n_epochs=train_epoch, batch_size=batch_size)
        classifier.learning_rate = tune_lr
        classifier.tune(train_data, valid_data=valid_data, batch_size=tune_batch, n_epochs=tune_epoch)
        scr = classifier.get_score(X_test)
        array = np.column_stack((scr[:, 1], R_test, G_test, Y_test))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'R', 'G', 'Y'])
        df = df.append(df_temp)

    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    A_CI = roc_auc_score(y_test, y_scr, average='weighted')
    res = {'TL_Auc': A_CI}
    
    df = pd.DataFrame(res, index=[seed])
    
    return df
    
def run_naive_transfer_cv(seed, Source, Target, dataset, groups, genders,
                        data_Category, omics_feature, FeatureMethod,
                        fold=3, k=-1, val_size=0,
                        batch_size=32, momentum=0.9,
                        learning_rate=0.01, lr_decay=0.0, 
                        dropout=0.5, n_epochs=100,
                        save_to=None, L1_reg=0.001,
                        L2_reg=0.001, hiddenLayers=[128, 64]):
    
    X, Y, R, y_strat, G, Gy_strat, GRy_strat = dataset
    
    if data_Category == "GenderRace":
        temp1 = genders[1] + '0' + groups[0] # 'FEMALE0WHITE'
        temp2 = genders[1] + '1' + groups[0] # 'FEMALE1WHITE'
        temp3 = genders[0] + '0' + groups[0] # 'MALE0WHITE'
        temp4 = genders[0] + '1' + groups[0] # 'MALE1WHITE'
        temp5 = genders[1] + '0' + groups[1] # 'FEMALE0DDP'
        temp6 = genders[1] + '1' + groups[1] # 'FEMALE1DDP'
        temp7 = genders[0] + '0' + groups[1] # 'MALE0DDP'
        temp8 = genders[0] + '1' + groups[1] # 'MALE1DDP'
    
    # Initialize DataFrame for results
    if len(np.shape(omics_feature)) == 0:  # single omics
        m = X.shape[1] if k < 0 else min(X.shape[1], k)
        columns = list(range(m))
    else:  # multi omics
        m_list = []
        for feature in omics_feature:
            f_num = feature_counts[feature]
            m_list.append(min(f_num, k) if k > 0 else f_num)
        m = sum(m_list)
        columns = list(range(m))
    
    columns.extend(['scr', 'R', 'G', 'Y'])
    df = pd.DataFrame(columns=columns)
    
    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    
    for train_index, test_index in kf.split(X, GRy_strat):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        R_train, R_test = R[train_index], R[test_index]
        y_strat_train, y_strat_test = y_strat[train_index], y_strat[test_index]
        G_train, G_test = G[train_index], G[test_index]
        Gy_strat_train, Gy_strat_test = Gy_strat[train_index], Gy_strat[test_index]
        GRy_strat_train, GRy_strat_test = GRy_strat[train_index], GRy_strat[test_index]
        
        if FeatureMethod == 0:  # ANOVA based feature selection
            if k > 0:  # k = features_count from the main function provided by user
                if len(np.shape(omics_feature)) == 0:  # single omics
                    if X_train.shape[1] > k:  # Check if the number of features in the training set is greater than k
                        print(f"Performing ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train.shape[1]}) in the training set is more than k ({k}).")
                        k_best = SelectKBest(f_classif, k=k)
                        k_best.fit(X_train, Y_train)
                        X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)
                    else:
                        print(f"Skipping ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train.shape[1]}) in the training set is less than or equal to k ({k}).")
                        # Skip feature selection and use all features
                        X_train, X_test = X_train, X_test
                else:  # multi omics
                    k_best_list = []
                    start_idx = 0
                    for feature in omics_feature:
                        f_num = feature_counts[feature]
                        end_idx = start_idx + f_num
                        actual_feature_count = X_train[:, start_idx:end_idx].shape[1]
                        if k < f_num:  # Check if k is less than the number of features in the dataset
                            print(f"Performing ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is more than k ({k}).")
                            k_best = SelectKBest(f_classif, k=k)
                            k_best.fit(X_train[:, start_idx:end_idx], Y_train)
                            k_best_list.append(k_best)
                        else:  # No feature selection needed
                            print(f"Skipping ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is less than or equal to k ({k}).")
                            k_best_list.append(None)
                        start_idx = end_idx
                    
                    # Transform and concatenate the features
                    X_train_transformed = []
                    X_test_transformed = []
                    start_idx = 0
                    for i, feature in enumerate(omics_feature):
                        f_num = feature_counts[feature]
                        end_idx = start_idx + f_num
                        if k_best_list[i] is not None:  # If feature selection was applied
                            X_train_transformed.append(k_best_list[i].transform(X_train[:, start_idx:end_idx]))
                            X_test_transformed.append(k_best_list[i].transform(X_test[:, start_idx:end_idx]))
                        else:  # No feature selection, use original data
                            X_train_transformed.append(X_train[:, start_idx:end_idx])
                            X_test_transformed.append(X_test[:, start_idx:end_idx])
                        start_idx = end_idx
                    
                    X_train = np.concatenate(X_train_transformed, axis=1)
                    X_test = np.concatenate(X_test_transformed, axis=1)
        
        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val, R_train, R_val, G_train, G_val, \
                y_strat_train, y_strat_val, Gy_strat_train, Gy_strat_val, \
                GRy_strat_train, GRy_strat_val = \
                train_test_split(X_train, Y_train, R_train, y_strat_train, \
                                 G_train, Gy_strat_train, GRy_strat_train)
            valid_data = (X_val, Y_val)
            
            if Source=='WHITE':
                idx = (R_val==groups[0])
            elif Source=='DDP':
                idx = (R_val==groups[1])
            elif Source=='MALE':
                idx = (G_val==genders[0])
            elif Source=='FEMALE':
                idx = (G_val==genders[1])              
            elif Source=='WHITE-FEMALE':
                idx = ((GRy_strat_val==temp1)|(GRy_strat_val==temp2))
            elif Source=='WHITE-MALE':
                idx = ((GRy_strat_val==temp3)|(GRy_strat_val==temp4))
            elif Source=='MG-FEMALE':
                idx = ((GRy_strat_val==temp5)|(GRy_strat_val==temp6))
            elif Source=='MG-MALE':
                idx = ((GRy_strat_val==temp7)|(GRy_strat_val==temp8))
            
            X_val, Y_val = X_val[idx==True], Y_val[idx==True]
            valid_data = (X_val, Y_val)
        
        if Source=='WHITE':
            idx = (R_train==groups[0])
        elif Source=='DDP':
            idx = (R_train==groups[1])
        elif Source=='FEMALE':
            idx = (G_train==genders[1])
        elif Source=='MALE':
            idx = (G_train==genders[0])
        elif Source=='WHITE-FEMALE':
            idx = ((GRy_strat_train==temp1)|(GRy_strat_train==temp2))
        elif Source=='WHITE-MALE':
            idx = ((GRy_strat_train==temp3)|(GRy_strat_train==temp4))
        elif Source=='DDP-FEMALE':
            idx = ((GRy_strat_train==temp5)|(GRy_strat_train==temp6))
        elif Source=='DDP-MALE':
            idx = ((GRy_strat_train==temp7)|(GRy_strat_train==temp8))
        
        X_train, Y_train = X_train[idx==True], Y_train[idx==True] 
        train_data = (X_train, Y_train)
        
        n_in = X_train.shape[1]
        classifier = MLP(n_in=n_in, learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout, L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers, momentum=momentum)
        classifier.train(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        
        if Target=='DDP-FEMALE':
            idx = ((GRy_strat_test==temp5)|(GRy_strat_test==temp6))
        elif Target=='DDP-MALE':
            idx = ((GRy_strat_test==temp7)|(GRy_strat_test==temp8))
        elif Target=='FEMALE':
            idx = (G_test==genders[1])
        elif Target=='MALE':
            idx = (G_test==genders[0])
        elif Target=='DDP':
            idx = (R_test==groups[1])
        
        X_test, Y_test, R_test, y_strat_test, G_test, Gy_strat_test, GRy_strat_test = \
            X_test[idx], Y_test[idx], R_test[idx], y_strat_test[idx], G_test[idx], Gy_strat_test[idx], GRy_strat_test[idx]
        X_scr = classifier.get_score(X_test)
        array1 = np.column_stack((X_test, X_scr[:,1], R_test, G_test, Y_test))
        df_temp1 = pd.DataFrame(array1, columns=columns)
        df = df.append(df_temp1)

    if save_to:
        df.to_csv(save_to)
    
    y_test_b, y_scr_b = list(df['Y'].values), list(df['scr'].values)
    B_CI = roc_auc_score(y_test_b, y_scr_b, average='weighted')
    res = {'NT_Auc': B_CI}
    df = pd.DataFrame(res, index=[seed])
    
    return df

## FADA (torch):
def FADA_classification(seed, Source, Target, dataset, groups, genders,
                      checkpt_path, data_Category, omics_feature, FeatureMethod, n_features,
                      fold=3, alpha=0.3, learning_rate = 0.01,
                      hiddenLayers=[128, 64], dr=0.5,
                      momentum=0.9, decay=0.0, batch_size=20,
                      sample_per_class=2, EarlyStop=False, DCD_optimizer='SGD',
                      patience=100, n_epochs=100,
                      L1_reg=0.001, L2_reg=0.001):
    
    valid_data = True if EarlyStop==True else None
    X, Y, R, y_strat, G, Gy_strat, GRy_strat = dataset
    
    # Initialize DataFrame for results
    if len(np.shape(omics_feature)) == 0:  # single omics
        m = X.shape[1] if n_features < 0 else min(X.shape[1], n_features)
    else:  # multi omics
        m_list = []
        for feature in omics_feature:
            f_num = feature_counts[feature]
            m_list.append(min(f_num, n_features) if n_features > 0 else f_num)
        m = sum(m_list)

    df = pd.DataFrame(X)
    df['R'] = R
    df['Y'] = Y
    df['G'] = G
    df['GRY'] = GRy_strat

    if data_Category == "GenderRace":
        temp1 = genders[1] + '0' + groups[0] # 'FEMALE0WHITE'
        temp2 = genders[1] + '1' + groups[0] # 'FEMALE1WHITE'
        temp3 = genders[0] + '0' + groups[0] # 'MALE0WHITE'
        temp4 = genders[0] + '1' + groups[0] # 'MALE1WHITE'
        temp5 = genders[1] + '0' + groups[1] # 'FEMALE0DDP'
        temp6 = genders[1] + '1' + groups[1] # 'FEMALE1DDP'
        temp7 = genders[0] + '0' + groups[1] # 'MALE0DDP'
        temp8 = genders[0] + '1' + groups[1] # 'MALE1DDP'
    
    # domain adaptation - source groups
    if Source=='WHITE':
        df_train = df[df['R']==groups[0]]
    elif Source=='DDP':
        df_train = df[df['R']==groups[1]]
    elif Source=='FEMALE':
        df_train = df[df['G']==genders[1]]
    elif Source=='MALE':
        df_train = df[df['G']==genders[0]]
    elif Source=='WHITE-FEMALE':
        df_train = df[df['GRY'].isin([temp1,temp2])]
    elif Source=='WHITE-MALE':
        df_train = df[df['GRY'].isin([temp3,temp4])]
    elif Source=='DDP-FEMALE':
        df_train = df[df['GRY'].isin([temp5,temp6])]
    elif Source=='DDP-MALE':
        df_train = df[df['GRY'].isin([temp7,temp8])]
    
    df_w_y = df_train['Y']
    df_train = df_train.drop(columns=['Y', 'R', 'G', 'GRY'])
    Y_train_source = df_w_y.values.ravel()
    X_train_source = df_train.values
    
 	#test groups
    if Target=='DDP':
        df_test = df[df['R']==groups[1]]
    elif Target=='DDP-FEMALE':
        df_test = df[df['GRY'].isin([temp5,temp6])]
    elif Target=='DDP-MALE':
        df_test = df[df['GRY'].isin([temp7,temp8])]
    elif Target=='FEMALE':
        df_test = df[df['G']==genders[1]]
    elif Target=='MALE':
        df_test = df[df['G']==genders[0]]
    
    df_b_y = df_test['Y']
    df_b_R = df_test['R'].values.ravel()
    df_b_G = df_test['G'].values.ravel()
    df_test = df_test.drop(columns=['Y', 'R', 'G', 'GRY'])
    Y_test = df_b_y.values.ravel()
    X_test = df_test.values
    
    if FeatureMethod == 0: # ANOVA based feature selection
        if n_features > 0: # n_features = features_count from the main function provided by user
            if len(np.shape(omics_feature)) == 0:  # single omics
                if X_train_source.shape[1] > n_features:  # Check if the number of features in the training set is greater than n_features
                    print(f"Performing ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train_source.shape[1]}) in the training set is more than n_features ({n_features}).")
                    X_train_source, X_test = get_k_best(X_train_source, Y_train_source, X_test, n_features)
                else:
                    print(f"Skipping ANOVA based feature selection for single omics {omics_feature} as the number of features ({X_train_source.shape[1]}) in the training set is less than or equal to n_features ({n_features}).")
                    # Skip feature selection and use all features
                    X_train_source, X_test = X_train_source, X_test
            else:  # multi omics
                k_best_list = []
                start_idx = 0
                for feature in omics_feature:
                    f_num = feature_counts[feature]
                    end_idx = start_idx + f_num
                    actual_feature_count = X_train_source[:, start_idx:end_idx].shape[1]
                    if n_features < f_num:  # Check if n_features is less than the number of features in the dataset
                        print(f"Performing ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is more than n_features ({n_features}).")
                        k_best = SelectKBest(f_classif, k=n_features)
                        k_best.fit(X_train_source[:, start_idx:end_idx], Y_train_source)
                        k_best_list.append(k_best)
                    else:  # No feature selection needed
                        print(f"Skipping ANOVA based feature selection for multi omics {feature} as the number of features ({actual_feature_count}) in the training set is less than or equal to n_features ({n_features}).")
                        k_best_list.append(None)
                    start_idx = end_idx
                
                # Transform and concatenate the features
                X_train_source_transformed = []
                X_test_transformed = []
                start_idx = 0
                for i, feature in enumerate(omics_feature):
                    f_num = feature_counts[feature]
                    end_idx = start_idx + f_num
                    if k_best_list[i] is not None:
                        X_train_source_transformed.append(k_best_list[i].transform(X_train_source[:, start_idx:end_idx]))
                        X_test_transformed.append(k_best_list[i].transform(X_test[:, start_idx:end_idx]))
                    else:
                        X_train_source_transformed.append(X_train_source[:, start_idx:end_idx])
                        X_test_transformed.append(X_test[:, start_idx:end_idx])
                    start_idx = end_idx
                
                X_train_source = np.concatenate(X_train_source_transformed, axis=1)
                X_test = np.concatenate(X_test_transformed, axis=1)
                
        else:
            n_features = X_test.shape[1]
    
    if sample_per_class==None:
        samples_provided = 'No'
    else: 
        samples_provided = 'Yes'
        
    trainData = dataloader.CSVDataset(X_train_source,Y_train_source)
    train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
    X_s = torch.tensor(X_train_source)
    Y_s = torch.tensor(Y_train_source,dtype=torch.int64)
    
    df_score = pd.DataFrame(columns=['scr', 'Y', 'R', 'G'])
    kf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(X_test, Y_test):
        X_train_target_full, X_test_target = X_test[train_index], X_test[test_index]
        Y_train_target_full, Y_test_target = Y_test[train_index], Y_test[test_index]
        R_train_target_full, R_test_target = df_b_R[train_index], df_b_R[test_index]
        G_train_target_full, G_test_target = df_b_G[train_index], df_b_G[test_index]
        if samples_provided=='No':
            maxallowedsamples_un = np.unique(Y_train_target_full,return_counts=True)
            maxallowedsamples = min(maxallowedsamples_un[1])
            # print('Max samples are allowed: '+ str(maxallowedsamples))
            X_train1,X_val1,Y_train1,Y_val1 = train_test_split(X_train_target_full,Y_train_target_full,random_state=None)
            target_samples_count = np.unique(Y_train1,return_counts=True)
            min_target_samples_count = min(target_samples_count[1])
            sample_per_class = min_target_samples_count
            # print('Sample per class is : '+str(sample_per_class))
            if sample_per_class==1:
                sample_per_class = 2
            elif sample_per_class>2:
                if sample_per_class>maxallowedsamples:
                    sample_per_class = maxallowedsamples
        # print('==================================')
        # print('Sample per class is : '+str(sample_per_class))
        # print('==================================')
        
        train_targetData = dataloader.CSVDataset(X_train_target_full,Y_train_target_full)
        X_t,Y_t,X_val,Y_val,valid_data = dataloader.create_target_samples_cancer(train_targetData,sample_per_class)
        
        if valid_data==True:
            #Y_val = F.one_hot(Y_val.to(torch.int64), num_classes=nb_classes)
            val_dataloader = dataloader.CSVDataset(X_val,Y_val)
            val_dataloader = DataLoader(val_dataloader,batch_size=len(X_val))
                
        net = fada_model.Network(in_features_data=m,nb_classes=2,dropout=dr,hiddenLayers=[128,64])
        net.to(device)
        loss_fn = torch.nn.BCELoss()
        
        discriminator = fada_model.DCD(h_features=64,input_features=128)#128=64*2 i.e. twice the output of classifier --> stacking
        discriminator.to(device)
        loss_discriminator = torch.nn.CrossEntropyLoss()
        
        #STEP 1:
        if DCD_optimizer=='SGD':
            optimizer1 = torch.optim.SGD(list(net.parameters()),lr=learning_rate,momentum=momentum)
        elif DCD_optimizer=='Adam':
            optimizer1 = torch.optim.Adam(list(net.parameters()),lr=learning_rate)
        #STEP 2:
        if DCD_optimizer=='SGD':
            optimizer_D = torch.optim.SGD(discriminator.parameters(),lr=learning_rate,momentum=momentum)
        elif DCD_optimizer=='Adam':
            optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=learning_rate)
        #STEP 3:
        if DCD_optimizer=='SGD':
            optimizer_g_h1 = torch.optim.SGD(list(net.parameters()),lr=learning_rate,momentum=momentum)
        elif DCD_optimizer=='Adam':
            optimizer_g_h1 = torch.optim.Adam(list(net.parameters()),lr=learning_rate)
        if DCD_optimizer=='SGD':
            optimizer_d = torch.optim.SGD(discriminator.parameters(),lr=learning_rate,momentum=momentum)
        elif DCD_optimizer=='Adam':
            optimizer_d = torch.optim.Adam(discriminator.parameters(),lr=learning_rate)
        
        # optimizer1 = torch.optim.SGD(list(net.parameters()),lr=learning_rate,momentum=momentum)
        optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=learning_rate)
        # optimizer_g_h1 = torch.optim.SGD(list(net.parameters()),lr=learning_rate,momentum=momentum)
        optimizer_d = torch.optim.Adam(discriminator.parameters(),lr=learning_rate)
        
        ###################
        # STEP 1:
        ###################
        train_losses_s1 = []
        avg_train_losses_s1 = []
        valid_losses_s1 = []
        valid_loss_s1 = []
        avg_valid_losses_s1 = []
        
        # initialize the early_stopping object
        early_stopping1 = EarlyStopping(patience=patience,verbose=True,path=checkpt_path)
        for epoch in range(n_epochs):
            for data,labels in train_dataloader:
                data = data.to(device)
                labels = labels.to(device)
                labels = labels.to(torch.long)
                #labels = F.one_hot(labels.to(torch.int64), num_classes=nb_classes)
                optimizer1.zero_grad()
                y_pred,_ = net(data)
                loss = loss_fn(y_pred[:,1],labels.float())
                l1_norm = sum(p.abs().sum() for p in net.parameters())
                l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                loss = loss + L2_reg * l2_norm + L1_reg * l1_norm
                loss.backward()
                optimizer1.step()
                train_losses_s1.append(loss.item())
                
            ######################    
            # validate the model #
            ######################
            if valid_data==True:
                with torch.no_grad():
                    for val_data,val_targets in val_dataloader:
                        val_data = val_data.to(device)
                        val_targets = val_targets.to(device)
                        val_targets = val_targets.to(torch.long)
                        val_pred,_ = net(val_data)
                        v_loss = loss_fn(val_pred[:,1],val_targets.float())
                        valid_losses_s1.append(v_loss.numpy())
            
            # print training/validation statistics
            train_loss_s1 = np.average(train_losses_s1)
            avg_train_losses_s1.append(train_loss_s1)
            if valid_data==True:
                valid_loss_s1 = np.average(valid_losses_s1)
                avg_valid_losses_s1.append(valid_loss_s1)
            
            epoch_len = len(str(n_epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' + f'train_loss_s1: {train_loss_s1:.5f} ')
            #print(print_msg)
            
            # clear lists to track next epoch
            train_losses_s1 = []
            valid_losses_s1 = []
            
            if valid_data==True:
                if EarlyStop==True:
                    early_stopping1(valid_loss_s1, net)
                    if early_stopping1.early_stop:
                        print("Early stopping")
                        break
        if valid_data==True:
            if EarlyStop==True:
                # load the last checkpoint with the best model
                net.load_state_dict(torch.load(checkpt_path))
        
        ###################
        # STEP 2:
        ###################
        train_losses_s2 = []
        avg_train_losses_s2 = []
        valid_losses_s2 = []
        valid_loss_s2 = []
        avg_valid_losses_s2 = []
        
        # initialize the early_stopping object
        early_stopping2 = EarlyStopping(patience=patience,verbose=True,path=checkpt_path)
        for epoch in range(n_epochs):
            groups,aa = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=n_epochs+epoch)
            n_iters = 4 * len(groups[1])
            index_list = torch.randperm(n_iters)
            mini_batch_size = n_iters//2
            X1 = [];
            X2 = [];
            ground_truths = []
            for index in range(n_iters):
                #ground_truth=index_list[index]//len(groups[1])
                ground_truth = torch.div(index_list[index],len(groups[1]),rounding_mode='floor')
                x1,x2 = groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]
                X1.append(x1)
                X2.append(x2)
                ground_truths.append(ground_truth)
        
                #select data for a mini-batch to train
                if (index+1)%mini_batch_size==0:
                    X1 = torch.stack(X1)
                    X2 = torch.stack(X2)
                    ground_truths = torch.LongTensor(ground_truths)
                    X1 = X1.to(device)
                    X2 = X2.to(device)
                    ground_truths = F.one_hot(ground_truths.to(torch.int64), num_classes=4)#4 groups
                    ground_truths = ground_truths.to(device)
                    optimizer_D.zero_grad()
                    _,feature_out1 = net(X1)
                    _,feature_out2 = net(X2)
                    X_cat = torch.cat([feature_out1,feature_out2],1)
                    y_pred = discriminator(X_cat.detach())
                    loss = loss_discriminator(y_pred,ground_truths.float())
                    l1_norm = sum(p.abs().sum() for p in discriminator.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in discriminator.parameters())
                    loss = loss + L2_reg * l2_norm + L1_reg * l1_norm
                    loss.backward()
                    optimizer_D.step()
                    train_losses_s2.append(loss.item())
                    
                    X1 = []
                    X2 = []
                    ground_truths = []
            
            ######################    
            # validate the model #
            ######################
            if valid_data==True:
                with torch.no_grad():
                    for val_data,val_targets in val_dataloader:
                        val_data = val_data.to(device)
                        val_targets = val_targets.to(device)
                        val_targets = val_targets.to(torch.long)
                        val_pred,_ = net(val_data)
                        v_loss = loss_fn(val_pred[:,1],val_targets.float())
                        valid_losses_s2.append(v_loss.numpy())
            
            # print training/validation statistics
            train_loss_s2 = np.average(train_losses_s2)
            avg_train_losses_s2.append(train_loss_s2)
            if valid_data==True:
                valid_loss_s2 = np.average(valid_losses_s2)
                avg_valid_losses_s2.append(valid_loss_s2)
            
            epoch_len = len(str(n_epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' + f'train_loss_s2: {train_loss_s2:.5f} ')
            #print(print_msg)
            
            # clear lists to track next epoch
            train_losses_s2 = []
            valid_losses_s2 = []
            
            if valid_data==True:
                if EarlyStop==True:
                    early_stopping2(valid_loss_s2, discriminator)
                    if early_stopping2.early_stop:
                        print("Early stopping")
                        break
        
        if valid_data==True:
            if EarlyStop==True:
                # load the last checkpoint with the best model
                discriminator.load_state_dict(torch.load(checkpt_path))
        
        ###################
        # STEP 3:
        ###################
        train_losses_gh = []
        train_losses_dcd = []
        valid_losses = []
        valid_loss = []
        avg_train_losses_gh = []
        avg_train_losses_dcd = []
        avg_valid_losses = [] 
        
        # initialize the early_stopping object
        early_stopping3 = EarlyStopping(patience=patience,verbose=True,path=checkpt_path)
        for epoch in range(n_epochs):
            #---training g and h , DCD is frozen        
            groups, groups_y = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=n_epochs+epoch)
            G1, G2, G3, G4 = groups
            Y1, Y2, Y3, Y4 = groups_y
            groups_2 = [G2, G4]
            groups_y_2 = [Y2, Y4]
            n_iters = 2 * len(G2)
            index_list = torch.randperm(n_iters)
            n_iters_dcd = 4 * len(G2)
            index_list_dcd = torch.randperm(n_iters_dcd)
            mini_batch_size_g_h = n_iters//2
            mini_batch_size_dcd = n_iters_dcd//2
            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []
            for index in range(n_iters):
                #ground_truth=index_list[index]//len(G2)
                ground_truth = torch.div(index_list[index],len(G2),rounding_mode='floor')
                x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
                y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
                dcd_label = 0 if ground_truth==0 else 2
                X1.append(x1)
                X2.append(x2)
                ground_truths_y1.append(y1)
                ground_truths_y2.append(y2)
                dcd_labels.append(dcd_label)
                if (index+1)%mini_batch_size_g_h==0:
                    X1 = torch.stack(X1)
                    X2 = torch.stack(X2)
                    ground_truths_y1 = torch.LongTensor(ground_truths_y1)
                    ground_truths_y2 = torch.LongTensor(ground_truths_y2)
                    dcd_labels = torch.LongTensor(dcd_labels)
                    X1 = X1.to(device)
                    X2 = X2.to(device)
                    #ground_truths_y1 = F.one_hot(ground_truths_y1.to(torch.int64), num_classes=2)
                    #ground_truths_y2 = F.one_hot(ground_truths_y2.to(torch.int64), num_classes=2)
                    ground_truths_y1 = ground_truths_y1.to(device)
                    ground_truths_y2 = ground_truths_y2.to(device)
                    dcd_labels = F.one_hot(dcd_labels.to(torch.int64), num_classes=4)
                    dcd_labels = dcd_labels.to(device)
                    optimizer_g_h1.zero_grad()
                    y_pred_X1,encoder_X1 = net(X1)
                    y_pred_X2,encoder_X2 = net(X2)
                    X_cat = torch.cat([encoder_X1,encoder_X2],1)
                    y_pred_dcd = discriminator(X_cat)
                    loss_X1 = loss_fn(y_pred_X1[:,1],ground_truths_y1.float())
                    l1_norm = sum(p.abs().sum() for p in net.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                    loss_X1 = loss_X1 + L2_reg * l2_norm + L1_reg * l1_norm
                    loss_X2 = loss_fn(y_pred_X2[:,1],ground_truths_y2.float())
                    l1_norm = sum(p.abs().sum() for p in net.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                    loss_X2 = loss_X2 + L2_reg * l2_norm + L1_reg * l1_norm
                    loss_dcd = loss_discriminator(y_pred_dcd,dcd_labels.float())
                    l1_norm = sum(p.abs().sum() for p in discriminator.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in discriminator.parameters())
                    loss_dcd = loss_dcd + L2_reg * l2_norm + L1_reg * l1_norm
                    loss_sum = (loss_X1 + loss_X2 + alpha*loss_dcd)/3
                    # l1_norm = sum(p.abs().sum() for p in net.parameters())
                    # l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                    # loss_sum = loss_sum + L2_reg * l2_norm + L1_reg * l1_norm
                    
                    loss_sum.backward()
                    optimizer_g_h1.step()
                    train_losses_gh.append(loss_sum.item())
        
                    X1 = []
                    X2 = []
                    ground_truths_y1 = []
                    ground_truths_y2 = []
                    dcd_labels = []
            
            #----training dcd ,g and h frozen
            X1 = []
            X2 = []
            ground_truths = []
            for index in range(n_iters_dcd):
                #ground_truth=index_list_dcd[index]//len(groups[1])
                ground_truth = torch.div(index_list_dcd[index],len(groups[1]),rounding_mode='floor')
                x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
                X1.append(x1)
                X2.append(x2)
                ground_truths.append(ground_truth)
        
                if (index + 1) % mini_batch_size_dcd == 0:
                    X1 = torch.stack(X1)
                    X2 = torch.stack(X2)
                    ground_truths = torch.LongTensor(ground_truths)
                    X1 = X1.to(device)
                    X2 = X2.to(device)
                    ground_truths = F.one_hot(ground_truths.to(torch.int64), num_classes=4)#4 groups
                    ground_truths = ground_truths.to(device)
                    optimizer_d.zero_grad()
                    _,feature_out11 = net(X1)
                    _,feature_out12 = net(X2)
                    X_cat = torch.cat([feature_out11, feature_out12], 1)
                    y_pred = discriminator(X_cat.detach())
                    loss = loss_discriminator(y_pred, ground_truths.float())
                    l1_norm = sum(p.abs().sum() for p in discriminator.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in discriminator.parameters())
                    loss = loss + L2_reg * l2_norm + L1_reg * l1_norm
                    loss.backward()
                    optimizer_d.step()
                    train_losses_dcd.append(loss.item())
                    
                    X1 = []
                    X2 = []
                    ground_truths = []
            
            ######################    
            # validate the model #
            ######################
            if valid_data==True:
                with torch.no_grad():
                    for val_data,val_targets in val_dataloader:
                        val_data = val_data.to(device)
                        val_targets = val_targets.to(device)
                        val_targets = val_targets.to(torch.long)
                        val_pred,_ = net(val_data)
                        v_loss = loss_fn(val_pred[:,1],val_targets.float())
                        valid_losses.append(v_loss.numpy())
            
            # print training/validation statistics 
            # calculate average loss over an epoch
            avg_train_losses_gh.append(np.average(train_losses_gh))
            avg_train_losses_dcd.append(np.average(train_losses_dcd))
            if valid_data==True:
                valid_loss = np.average(valid_losses)
                avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(n_epochs))
            train_loss_gh = np.average(train_losses_gh)
            train_loss_dcd = np.average(train_losses_dcd)
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                          f'train_losses_gh: {train_loss_gh:.5f} ' +
                          f'train_losses_dcd: {train_loss_dcd:.5f} ')
            
            #print(print_msg)
            
            # clear lists to track next epoch
            train_losses_gh = []
            train_losses_dcd = []
            valid_losses = []
            
            if valid_data==True:
                if EarlyStop==True:
                    # early_stopping needs the validation loss to check if it has decresed, 
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping3(valid_loss, net)
                    if early_stopping3.early_stop:
                        print("Early stopping")
                        break
        
        if valid_data==True:
            if EarlyStop==True:
                # load the last checkpoint with the best model
                net.load_state_dict(torch.load(checkpt_path))
        
        ######################    
        # testing the model #
        ######################
        X_test_target = torch.tensor(X_test_target)
        X_test_target = X_test_target.to(device)
        Y_test_target = torch.tensor(Y_test_target)
        #Y_test_target = F.one_hot(Y_test_target.to(torch.int64), num_classes=nb_classes)
        Y_test_target = Y_test_target.to(device)
        
        with torch.no_grad():
            y_test_pred,_ = net(X_test_target)
            #_, idx = y_test_pred.max(dim=1)
            
        best_score = y_test_pred[:,1]
        array = np.column_stack((best_score, Y_test[test_index], R_test_target, G_test_target))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'Y', 'R', 'G'])
        df_score = df_score.append(df_temp)
    
    auc = roc_auc_score(list(df_score['Y'].values), list(df_score['scr'].values), average='weighted')
    res = {'TL_DCD_Auc': auc}
    
    df_DCD_TL = pd.DataFrame(res, index=[seed])
    # print(res)
    
    return df_DCD_TL
    