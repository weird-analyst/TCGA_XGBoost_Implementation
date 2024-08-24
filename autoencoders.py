# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:56:30 2024

@author: teesh
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers, Model, Input
from keras.models import model_from_json
from preProcess import get_protein, get_mRNA, get_MicroRNA, \
    get_Methylation, standarize_dataset
import numpy as np
# import pandas as pd

def load_or_train_encoders(omics_feature, folderISAAC, cancer_type, endpoint, 
                           groups, genders, data_Category, features_count, 
                           AE_iter, AE_batchsize, FeatureMethod,
                           EncActiv, DecActiv, loss_fn, AutoencoderSettings):
    
    if FeatureMethod != 2:
        return None
    
    # Initialize dictionaries to store model names and file paths
    AE_json_file_names = {}
    E_json_file_names = {}
    encoders = {}

    if isinstance(omics_feature, str):
        omics_feature = [omics_feature]
        
    for omics in omics_feature:
        AE_ModelName = 'TCGA-' + omics + '-' + str(features_count)
        AE_json_file_name = folderISAAC + 'AE_models/autoencoder_' + str(AutoencoderSettings) + '_' + AE_ModelName + '.json'
        AE_json_file_names[omics] = AE_json_file_name
        E_json_file_name = folderISAAC + 'AE_models/encoder_' + str(AutoencoderSettings) + '_' + AE_ModelName + '.json'
        E_json_file_names[omics] = E_json_file_name
        print(f'AE_ModelName for {omics}: {AE_ModelName}')
        print(f'AE json file name for {omics}: {AE_json_file_name}')
        print(f'Encoder json file name for {omics}: {E_json_file_name}')
    
        # Check existence of AE model files
        if os.path.exists(AE_json_file_names[omics]) and os.path.exists(E_json_file_names[omics]):
            print(f"AE model for {omics} exists. Loading the model.")
            with open(E_json_file_names[omics], 'r') as json_file:
                loaded_model_json = json_file.read()
            encoders[omics] = model_from_json(loaded_model_json)
        else:
            print(f"AE model for {omics} does not exist. Training the model.")
            if omics == 'mRNA':
                dataset = get_mRNA(cancer_type=cancer_type, endpoint=endpoint, 
                                groups=groups, genders=genders,
                                FeatureMethod=FeatureMethod, FeatureTrain=True)
            elif omics == 'MicroRNA':
                dataset = get_MicroRNA(cancer_type=cancer_type, endpoint=endpoint, 
                                groups=groups, genders=genders,
                                FeatureMethod=FeatureMethod, FeatureTrain=True)
            elif omics == 'Protein':
                dataset = get_protein(cancer_type=cancer_type, endpoint=endpoint, 
                                groups=groups, genders=genders,
                                FeatureMethod=FeatureMethod, FeatureTrain=True)
            elif omics == 'Methylation':
                dataset = get_Methylation(cancer_type=cancer_type, endpoint=endpoint, 
                                groups=groups, genders=genders,
                                FeatureMethod=FeatureMethod, FeatureTrain=True)
            
            dataset = standarize_dataset(dataset)
            X = dataset['X']
            print("The shape of the dataset for AE training is: "+str(np.shape(dataset['X'])))
            
            if X.shape[1] < features_count:
                print(f"Number of features in {omics} is less than the specified features_count. Using all features.")
                encoders[omics] = None  # No encoder needed
                continue
            
            encoding_dim = features_count

            print(f'Train AE for {omics} with {features_count} features')

            standard_scaler = MinMaxScaler()
            X_AE = pd.DataFrame(standard_scaler.fit_transform(X))

            encoded_input = Input(shape=(X_AE.shape[1],))
            encoded = layers.Dense(encoding_dim, activation=EncActiv, name='encoder')(encoded_input)
            decoded = layers.Dense(X_AE.shape[1], activation=DecActiv)(encoded)
            autoencoder = Model(encoded_input, decoded)
            autoencoder.compile(optimizer='adam', loss=loss_fn)

            autoencoder.fit(X_AE, X_AE, epochs=AE_iter, batch_size=AE_batchsize, shuffle=True)

            encoder = Model(encoded_input, encoded)

            json_model = autoencoder.to_json()
            with open(AE_json_file_names[omics], 'w') as json_file:
                json_file.write(json_model)
            json_model = encoder.to_json()
            with open(E_json_file_names[omics], 'w') as json_file:
                json_file.write(json_model)

            encoders[omics] = encoder

    return encoders
    
def process_omics(datasets, encoders, omics_feature):
    
    if isinstance(omics_feature, str):  # single omics
    
        X = datasets[omics_feature]['X']
        standard_scaler = MinMaxScaler()
        if encoders[omics_feature] is not None:
            X_AE = pd.DataFrame(standard_scaler.fit_transform(X))
            reduced_df_X_AE = pd.DataFrame(encoders[omics_feature].predict(np.array(X_AE)))
            X_AE = np.array(reduced_df_X_AE)
            datasets[omics_feature]['X'] = X_AE
        else:
            print("No feature extraction is applied as encoder is None due to less number of features than provided by the user.")
        
        data_out = datasets[omics_feature]
        
    else:  # multi omics
        
        # Step 1: Find common sample IDs across all omics datasets
        sample_ids_sets = [set(datasets[omics]['Samples']) for omics in omics_feature]
        common_sample_ids = list(set.intersection(*sample_ids_sets))  # Find common sample IDs
    
        scaled_dfs = []
        non_unique_rows = {}
    
        # Step 2: Align datasets based on common sample IDs, scale, and encode
        for omics in omics_feature:
            # Align datasets based on common sample IDs
            aligned_data = pd.DataFrame(datasets[omics]['X'], index=datasets[omics]['Samples'])
            aligned_data = aligned_data.loc[common_sample_ids]  # Align to the common sample IDs
    
            # Identify non-unique indices
            non_unique_indices = aligned_data.index[aligned_data.index.duplicated(keep=False)]
            non_unique_rows[omics] = aligned_data.loc[non_unique_indices]
    
            # Filter out non-unique rows from aligned_data
            aligned_data = aligned_data.loc[~aligned_data.index.duplicated(keep='first')]
    
            # Scale and encode
            standard_scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(standard_scaler.fit_transform(aligned_data), index=aligned_data.index)
    
            if encoders[omics] is not None:
                reduced_df_X_scaled = pd.DataFrame(encoders[omics].predict(np.array(X_scaled)), index=common_sample_ids)
                scaled_dfs.append(reduced_df_X_scaled)
            else:
                scaled_dfs.append(X_scaled)
    
        # Step 3: Combine all scaled data into a single dataset
        X_AE = pd.concat(scaled_dfs, axis=1).values

        combined_dataset = {'X': X_AE, 'Samples': common_sample_ids}
        datasets['Combined'] = combined_dataset

        # Step 4: Extract sample IDs from Combined dataset
        combined_sample_ids = datasets['Combined']['Samples']

        # Function to filter dataset based on combined sample IDs
        def filter_dataset(dataset, combined_sample_ids):
            # Find the indices of the combined sample IDs in the dataset Samples
            sample_indices = np.where(np.isin(dataset['Samples'], combined_sample_ids))[0]

            # Filter each array in the dataset based on these indices
            filtered_dataset = {}
            for key, value in dataset.items():
                if isinstance(value, np.ndarray) and value.shape[0] == len(dataset['Samples']):
                    filtered_dataset[key] = value[sample_indices]
                else:
                    filtered_dataset[key] = value
            return filtered_dataset

        # Filter datasets based on combined_sample_ids
        filtered_datasets = {omics: filter_dataset(datasets[omics], combined_sample_ids) for omics in omics_feature}

        # Function to compare arrays and identify differing rows
        def compare_arrays(arr1, arr2):
            if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
                if np.array_equal(arr1, arr2):
                    return True, []
                else:
                    differing_indices = np.where(arr1 != arr2)[0]
                    return False, differing_indices.tolist()
            return False, []

        # List of keys to compare
        keys_to_compare = ['C', 'E', 'G', 'R', 'T']

        # Dictionary to store comparison results
        invalid_sample_indices = set()

        # Remove invalid samples
        valid_sample_indices = [i for i in range(len(common_sample_ids)) if i not in invalid_sample_indices]
        valid_common_sample_ids = [common_sample_ids[i] for i in valid_sample_indices]

        # Function to filter dataset to keep only valid samples
        def filter_valid_samples(dataset, valid_sample_indices):
            filtered_dataset = {}
            for key, value in dataset.items():
                if isinstance(value, np.ndarray) and value.shape[0] == len(dataset['Samples']):
                    filtered_dataset[key] = value[valid_sample_indices]
                else:
                    filtered_dataset[key] = value
            return filtered_dataset

        # Apply filtering to keep only valid samples
        datasets['Combined'] = {'X': X_AE[valid_sample_indices], 'Samples': valid_common_sample_ids}
        filtered_datasets = {omics: filter_valid_samples(filtered_datasets[omics], valid_sample_indices) for omics in omics_feature}
        
        # Create a new key in the dictionary named by joining the omics features
        new_key = "_".join(omics_feature)
        datasets[new_key] = {}
        for key in keys_to_compare + ['Samples']:
            datasets[new_key][key] = filtered_datasets[omics_feature[0]][key]  # Select any one dataset's key
        
        # Add X from the Combined key
        datasets[new_key]['X'] = datasets['Combined']['X']
        
        data_out = datasets[new_key]
        
    return data_out




