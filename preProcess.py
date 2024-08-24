# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:56:30 2024

@author: teesh
"""

from scipy.io import loadmat
import pandas as pd
import numpy as np
import os.path
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import sys
sys.stdout.flush()

path_to_data_DELL = 'C:/Users/DELL/OneDrive - Indian Institute of Technology Guwahati/Dataset/EssentialData/'
path_to_data_IITG = 'C:/Users/IITG/OneDrive - Indian Institute of Technology Guwahati/Dataset/EssentialData/'
path_to_data_IITG = 'C:/Users/teesh/OneDrive - Indian Institute of Technology Guwahati/Dataset/EssentialData/'

if os.path.exists(path_to_data_DELL)==True:
    DataPath = path_to_data_DELL
elif os.path.exists(path_to_data_IITG)==True:
    DataPath = path_to_data_IITG
else:
    DataPath = './Dataset/EssentialData/'

MethylationDataPath = os.path.join(DataPath, 'MethylationData/Methylation.mat')
MethyAncsDataPath = os.path.join(DataPath, 'MethylationData/MethylationGenetic.xlsx')
MethyCIDataPath = os.path.join(DataPath, 'MethylationData/MethylationClinInfo.xlsx')
MicroRNADataPath = os.path.join(DataPath, 'MicroRNAData/MicroRNA-Expression.mat')
mRNADataPath = os.path.join(DataPath, 'mRNAData/mRNA.mat')
ProteinDataPath = os.path.join(DataPath, 'ProteinData/Protein.txt')
GADataPath = os.path.join(DataPath, 'Genetic_Ancestry.xlsx')
OutcomeDataPath = os.path.join(DataPath, 'TCGA-CDR-SupplementalTableS1.xlsx')

def get_n_years(dataset, years):
    
    X, T, C = dataset['X'], dataset['T'], dataset['C']
    df = pd.DataFrame(X)
    df['T'] = T
    df['C'] = C
    df['Y'] = 1
    R, G = dataset['R'], dataset['G']
    df['R'] = R
    df['G'] = G
    df = df[~((df['T'] < 365 * years) & (df['C'] == 1))]
    df.loc[df['T'] <= 365 * years, 'Y'] = 0
    df['strat'] = df.apply(lambda row: str(row['Y']) + str(row['R']), axis=1)
    df['Gstrat'] = df.apply(lambda row: str(row['Y']) + str(row['G']), axis=1)
    df['GRstrat'] = df.apply(lambda row: str(row['G']) + str(row['Y']) + str(row['R']), axis=1)
    df = df.reset_index(drop=True)
    R = df['R'].values
    G = df['G'].values
    Y = df['Y'].values
    y_strat = df['strat'].values
    Gy_strat = df['Gstrat'].values
    GRy_strat = df['GRstrat'].values
    df = df.drop(columns=['T', 'C', 'R', 'G', 'Y', 'strat', 'Gstrat', 'GRstrat'])
    X = df.values
    
    return (X, Y.astype('int32'), R, y_strat, G, Gy_strat, GRy_strat)

def tumor_types(cancer_type):
    Map = {'GBMLGG': ['GBM', 'LGG'],
           'COADREAD': ['COAD', 'READ'],
           'KIPAN': ['KIRC', 'KICH', 'KIRP'],
           'STES': ['ESCA', 'STAD'],
           'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'],
           'PanGyn': ['OV', 'CESC', 'UCS', 'UCEC'],
           'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'],
           'PanPan': ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC',
                      'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG',
                      'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ',
                      'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
           }
    if cancer_type not in Map:
        Map[cancer_type] = [cancer_type]

    return Map[cancer_type]

def get_protein(cancer_type, endpoint, groups, genders, FeatureMethod, FeatureTrain):
    
    df = pd.read_csv(ProteinDataPath, sep='\t', index_col='SampleID').dropna(axis=1)
    if FeatureTrain: # AE or PCA
        if FeatureMethod in [1,2]:
            cancer_type = ['ACC','BLCA','BRCA','CESC',
                            'CHOL','COAD','DLBC','ESCA',
                            'GBM','HNSC','KICH','KIRC',
                            'KIRP','LAML','LGG','LIHC',
                            'LUAD','LUSC','MESO','OV',
                            'PAAD','PCPG','PRAD','READ',
                            'SARC','SKCM','STAD','TGCT',
                            'THCA','THYM','UCEC','UCS','UVM']
            tumorTypes = cancer_type
    else: # ANOVA or Original Features
        tumorTypes = tumor_types(cancer_type)
        
    df = df[df['TumorType'].isin(tumorTypes)]
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new

    return add_race_CT(tumorTypes, df, endpoint, groups, genders, FeatureMethod, FeatureTrain)

def get_Methylation(cancer_type, endpoint, groups, genders, FeatureMethod, FeatureTrain):
    
    MethylationData = loadmat(MethylationDataPath)
    X, Y, GeneName, SampleName = MethylationData['X'].astype('float32'), MethylationData['CancerType'], MethylationData['FeatureName'][0], MethylationData['SampleName']
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    MethylationData_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    MethylationData_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    if FeatureTrain: # AE or PCA
        if FeatureMethod in [1,2]:
            cancer_type = ['ACC','BLCA','BRCA','CESC',
                            'CHOL','COAD','DLBC','ESCA',
                            'GBM','HNSC','KICH','KIRC',
                            'KIRP','LAML','LGG','LIHC',
                            'LUAD','LUSC','MESO','OV',
                            'PAAD','PCPG','PRAD','READ',
                            'SARC','SKCM','STAD','TGCT',
                            'THCA','THYM','UCEC','UCS','UVM']
            tumorTypes = cancer_type
    else: # ANOVA or Original Features
        tumorTypes = tumor_types(cancer_type)
    
    MethylationData_Y = MethylationData_Y[MethylationData_Y['Disease'].isin(tumorTypes)]
    MethylationData_Y = MethylationData_Y.rename(columns={'Disease':'TumorType'})
    MethylationData_in = MethylationData_X.join(MethylationData_Y, how='inner')
    MethylationData_in.index = [row[:12] for row in MethylationData_in.index.values]
    MethylationData_in = MethylationData_in.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
    
    if FeatureTrain: # AE
        
        if FeatureMethod==2: # AE - Data for training
        
            MethylationData_in = MethylationData_in.dropna(axis='columns')
            MethylationData_in = MethylationData_in.drop(columns=['TumorType'])
            # Packing the data
            X = MethylationData_in.values
            X = MethylationData_in.astype('float32')
            InputData = {'X': X,
                         'Samples': MethylationData_in.index.values, 
                         'FeatureName': list(MethylationData_in)}
            
    else: # data for execution of ML task
    
        MethyAncsData = [
            pd.read_excel(MethyAncsDataPath, disease, usecols='A,B', index_col='bcr_patient_barcode', keep_default_na=False)
            for disease in tumorTypes
        ]
        GAData_race = pd.concat(MethyAncsData)
        GAData_race = GAData_race.rename(columns={'race': 'EIGENSTRAT'})
        GAData_race['race'] = GAData_race['EIGENSTRAT']
        GAData_race.loc[GAData_race['EIGENSTRAT'] == 'WHITE', 'race'] = 'WHITE'
        GAData_race.loc[GAData_race['EIGENSTRAT'] == 'BLACK OR AFRICAN AMERICAN', 'race'] = 'BLACK'
        GAData_race.loc[GAData_race['EIGENSTRAT'] == 'ASIAN', 'race'] = 'ASIAN'
        GAData_race.loc[GAData_race['EIGENSTRAT'] == 'AMERICAN INDIAN OR ALASKA NATIVE', 'race'] = 'NAT_A'
        GAData_race.loc[GAData_race['EIGENSTRAT'] == 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'race'] = 'OTHER'
        GAData_race = GAData_race.drop(columns=['EIGENSTRAT'])
        GAData_race = GAData_race[GAData_race['race'].isin(groups)]
    
        if endpoint == 'OS':
            cols = 'A,D,Y,Z'
        elif endpoint == 'DSS':
            cols = 'A,D,AA,AB'
        elif endpoint == 'DFI':
            cols = 'A,D,AC,AD'
        elif endpoint == 'PFI':
            cols = 'A,D,AE,AF'
        OutcomeData = pd.read_excel(MethyCIDataPath, usecols=cols, dtype={'OS': np.float64}, index_col='bcr_patient_barcode')
        
        OutcomeData = OutcomeData[OutcomeData['gender'].isin(genders)]
        OutcomeData.columns = ['G', 'E', 'T']
        OutcomeData = OutcomeData[OutcomeData['E'].isin([0, 1])].dropna()
        OutcomeData['C'] = 1 - OutcomeData['E']
        OutcomeData.drop(columns=['E'], inplace=True)
        MethylationData_in = MethylationData_in.join(GAData_race, how='inner').dropna(axis='columns')
        MethylationData_in = MethylationData_in.join(OutcomeData, how='inner')
        Data = MethylationData_in
        C = Data['C'].tolist()
        R = Data['race'].tolist()
        G = Data['G'].tolist()
        T = Data['T'].tolist()
        E = [1 - c for c in C]
        TumorType = Data['TumorType'].tolist()
        Data = Data.drop(columns=['C', 'race', 'T', 'G', 'TumorType'])
        X = Data.values.astype('float32')
        InputData = {'X': X, 
                     'T': np.asarray(T, dtype=np.float32), 
                     'C': np.asarray(C, dtype=np.int32), 
                     'E': np.asarray(E, dtype=np.int32), 
                     'R': np.asarray(R), 
                     'G': np.asarray(G), 
                     'Samples': Data.index.values, 
                     'FeatureName': list(Data),
                     'TumorType': list(TumorType)}
        
    return InputData

def get_mRNA(cancer_type, endpoint, groups, genders, FeatureMethod, FeatureTrain):
    
    A = loadmat(mRNADataPath)
    X, Y, GeneName, SampleName = A['X'].astype('float32'), A['Y'], A['GeneName'][0], A['SampleName']
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    if FeatureTrain: # AE or PCA
        if FeatureMethod in [1,2]:
            cancer_type = ['ACC','BLCA','BRCA','CESC',
                            'CHOL','COAD','DLBC','ESCA',
                            'GBM','HNSC','KICH','KIRC',
                            'KIRP','LAML','LGG','LIHC',
                            'LUAD','LUSC','MESO','OV',
                            'PAAD','PCPG','PRAD','READ',
                            'SARC','SKCM','STAD','TGCT',
                            'THCA','THYM','UCEC','UCS','UVM']
            tumorTypes = cancer_type
    else: # ANOVA or Original Features
        tumorTypes = tumor_types(cancer_type)
    
    df_Y = df_Y[df_Y['Disease'].isin(tumorTypes)]
    df = df_X.join(df_Y, how='inner')
    df = df.rename(columns={'Disease':'TumorType'})
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    return add_race_CT(tumorTypes, df, endpoint, groups, genders, FeatureMethod, FeatureTrain)

def get_MicroRNA(cancer_type, endpoint, groups, genders, FeatureMethod, FeatureTrain):
    
    A = loadmat(MicroRNADataPath)
    X, Y, GeneName, SampleName = A['X'].astype('float32'), A['CancerType'], A['FeatureName'][0], A['SampleName']
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    if FeatureTrain: # AE or PCA
        if FeatureMethod in [1,2]:
            cancer_type = ['ACC','BLCA','BRCA','CESC',
                            'CHOL','COAD','DLBC','ESCA',
                            'GBM','HNSC','KICH','KIRC',
                            'KIRP','LAML','LGG','LIHC',
                            'LUAD','LUSC','MESO','OV',
                            'PAAD','PCPG','PRAD','READ',
                            'SARC','SKCM','STAD','TGCT',
                            'THCA','THYM','UCEC','UCS','UVM']
            tumorTypes = cancer_type
    else: # ANOVA or Original Features
        tumorTypes = tumor_types(cancer_type)
    
    df_Y = df_Y[df_Y['Disease'].isin(tumorTypes)]
    df = df_X.join(df_Y, how='inner')
    df = df.rename(columns={'Disease':'TumorType'})
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    return add_race_CT(tumorTypes, df, endpoint, groups, genders, FeatureMethod, FeatureTrain)

def add_race_CT(tumorTypes, df, endpoint, groups, genders, FeatureMethod, FeatureTrain):
    
    if FeatureTrain: # AE
        
        if FeatureMethod==2: # AE - Data for training
        
            df = df.dropna(axis='columns')
            df = df.drop(columns=['TumorType'])
            # Packing the data
            X = df.values
            X = X.astype('float32')
            data = {'X': X,
                    'Samples': df.index.values,
                    'FeatureName': list(df)}
        
    else: # data for execution of ML task
    
        df_list = [pd.read_excel(GADataPath, disease, usecols='A,E', index_col='Patient_ID', keep_default_na=False)
                   for disease in tumorTypes]
        df_race = pd.concat(df_list)
        df_race = df_race[df_race['EIGENSTRAT'].isin(['EA', 'AA', 'EAA', 'NA', 'OA'])]
        df_race['race'] = df_race['EIGENSTRAT']
        df_race.loc[df_race['EIGENSTRAT'] == 'EA', 'race'] = 'WHITE'
        df_race.loc[df_race['EIGENSTRAT'] == 'AA', 'race'] = 'BLACK'
        df_race.loc[df_race['EIGENSTRAT'] == 'EAA', 'race'] = 'ASIAN'
        df_race.loc[df_race['EIGENSTRAT'] == 'NA', 'race'] = 'NAT_A'
        df_race.loc[df_race['EIGENSTRAT'] == 'OA', 'race'] = 'OTHER'
        df_race = df_race.drop(columns=['EIGENSTRAT'])
        
        df_race = df_race[df_race['race'].isin(groups)]
            
        cols = 'B,E,Z,AA'
        if endpoint == 'DSS':
            cols = 'B,E,AB,AC'
        elif endpoint == 'DFI':
            cols = 'B,E,AD,AE'
        elif endpoint == 'PFI':
            cols = 'B,E,AF,AG'
        df_C_T = pd.read_excel(OutcomeDataPath, 'TCGA-CDR', usecols=cols, index_col='bcr_patient_barcode')
        
        df_C_T.columns = ['G', 'E', 'T']
        df_C_T = df_C_T[df_C_T['G'].isin(genders)]
        df_C_T = df_C_T[df_C_T['E'].isin([0, 1])]
        df_C_T = df_C_T.dropna()
        df_C_T['C'] = 1 - df_C_T['E']
        df_C_T.drop(columns=['E'], inplace=True)
        
        df = df.join(df_race, how='inner')
        df = df.dropna(axis='columns')
        df = df.join(df_C_T, how='inner')
        
        # Packing the data
        C = df['C'].tolist()
        R = df['race'].tolist()
        G = df['G'].tolist()
        T = df['T'].tolist()
        E = [1 - c for c in C]
        TumorType = df['TumorType'].tolist()
        df = df.drop(columns=['C', 'race', 'T', 'G', 'TumorType'])
        X = df.values
        X = X.astype('float32')
        
        data = {'X': X,
                'T': np.asarray(T, dtype=np.float32),
                'C': np.asarray(C, dtype=np.int32),
                'E': np.asarray(E, dtype=np.int32),
                'R': np.asarray(R),
                'G': np.asarray(G),
                'Samples': df.index.values,
                'FeatureName': list(df),
                'TumorType': list(TumorType)}
    
    return data

def normalize_dataset(data):
    
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    X = preprocessing.normalize(X)
    data_new['X'] = X
    
    return data_new

def standarize_dataset(data):
    
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    data_new['X'] = X
    
    return data_new

def get_independent_data_single(dataset, criteria, query, groups, genders):
    
    # Unpack the dataset
    X, T, C, E, R, G = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R'], dataset['G']
    
    # Create a DataFrame from the feature matrix X and add R, G, and RG columns
    df = pd.DataFrame(X)
    df['R'] = R
    df['G'] = G
    df['RG'] = df['R'] + df['G']
    
    # # Ensure proper parameters based on criteria
    # if criteria == 'Gender':
    #     if query not in ['MALE', 'FEMALE']:
    #         raise ValueError("Invalid query parameter for Gender criteria")
    #     if genders is None:
    #         raise ValueError("Genders must be provided for Gender criteria")
    # elif criteria in ['Race', 'GenderRace']:
    #     if query not in ['WHITE', 'DDP', 'WHITE-FEMALE', 'WHITE-MALE', 'DDP-FEMALE', 'DDP-MALE']:
    #         raise ValueError("Invalid query parameter for Race or GenderRace criteria")
    #     if groups is None:
    #         raise ValueError("Groups must be provided for Race or GenderRace criteria")
    # else:
    #     raise ValueError("Invalid criteria parameter")
        
    # Define group-gender combinations if necessary
    if criteria in ['Race', 'GenderRace']:
        t1 = groups[0] + genders[1]  # WHITE-FEMALE
        t2 = groups[0] + genders[0]  # WHITE-MALE
        t3 = groups[1] + genders[1]  # DDP-FEMALE
        t4 = groups[1] + genders[0]  # DDP-MALE
    
    # Apply filters based on the criteria and query
    if query == 'MALE':
        mask = df['G'] == genders[0]
    elif query == 'FEMALE':
        mask = df['G'] == genders[1]
    elif query == 'WHITE':
        mask = df['R'] == groups[0]
    elif query == 'DDP':
        mask = df['R'] == groups[1]
    elif query == 'WHITE-FEMALE':
        mask = df['RG'] == t1
    elif query == 'WHITE-MALE':
        mask = df['RG'] == t2
    elif query == 'DDP-FEMALE':
        mask = df['RG'] == t3
    elif query == 'DDP-MALE':
        mask = df['RG'] == t4
    else:
        raise ValueError("Invalid query parameter")
    
    # Filter the dataset based on the mask
    X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
    
    # Pack the filtered data back into a dictionary
    data = {'X': X, 'T': T, 'C': C, 'E': E, 'R': R, 'G': G}
    
    return data

def merge_datasets(datasets):
    
    # Initialize an empty dataframe for merging
    df = pd.DataFrame()

    # Loop through each dataset and merge them based on common samples
    for key in datasets:
        data = datasets[key]
        X, Samples, FeatureName = data['X'], data['Samples'], data['FeatureName']
        temp_df = pd.DataFrame(X, index=Samples, columns=FeatureName)

        # If the main dataframe is empty, initialize it with the first dataset
        if df.empty:
            df = temp_df
            df['T'] = data['T']
            df['C'] = data['C']
            df['E'] = data['E']
            df['R'] = data['R']
            df['G'] = data['G']
        else:
            # Merge datasets on common samples
            df = df.join(temp_df, how='inner')

    # Extract the outcome variables and drop them from the main dataframe
    C = df['C'].tolist()
    R = df['R'].tolist()
    G = df['G'].tolist()
    T = df['T'].tolist()
    E = df['E'].tolist()
    df = df.drop(columns=['C', 'R', 'G', 'T', 'E'])

    # Convert the dataframe to a numpy array
    X = df.values.astype('float32')

    # Pack the data into a dictionary
    merged_data = {
        'X': X,
        'T': np.asarray(T, dtype=np.float32),
        'C': np.asarray(C, dtype=np.int32),
        'E': np.asarray(E, dtype=np.int32),
        'R': np.asarray(R),
        'G': np.asarray(G),
        'Samples': df.index.values,
        'FeatureName': list(df.columns)
    }

    return merged_data


