import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import torch
from sklearn.model_selection import train_test_split
from leaspy import Leaspy, Data, AlgorithmSettings
from scipy import stats


def id_based_train_test_split(df, val_size = 0.1, test_size=0.1, group_col='id', random_state=42):
    # Get unique subject IDs
    unique_subjects = df[group_col].unique()

    # Split the subject IDs into train and test groups
    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_state
    )

    train_subjects, val_subjects = train_test_split(
        train_subjects,
        test_size=val_size,
        random_state=random_state
    )

    # Create train and test dataframes based on the split subject IDs
    train_df = df[df[group_col].isin(train_subjects)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_subjects)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_subjects)].reset_index(drop=True)


    return train_df, val_df, test_df


def build_compatible_leaspy_dataframe(dataloader, model, device):
    with torch.no_grad():
        encodings = []
        times = []
        ids = []

        for images_tensor, id_list, timesteps_list in dataloader:
            # images_tensor: (T, 1, H, W)
            mu, _ = model.encoder(images_tensor.float().to(device))  # (T, latent_dim)
            encodings.append(mu.cpu())

            # Flatten timesteps and repeat IDs
            times.extend(timesteps_list.tolist())
            ids.extend([str(x) for x in id_list.tolist()])  # id_list est identique pour chaque pas de temps

        # Stack all encodings
        encodings_tensor = torch.cat(encodings, dim=0)  # (total_obs, latent_dim)

        # Convert to DataFrame
        encodings_df = pd.DataFrame(encodings_tensor.numpy(), columns=[f"ENCODING{i}" for i in range(encodings_tensor.shape[1])])
        encodings_df.insert(0, "TIME", times)
        encodings_df.insert(0, "ID", ids)
    return encodings_df




def statistical_test(xi, xi_mean, xi_std):
    """
    Teste si xi est dans la loi normale de moyenne xi_mean et d'écart-type xi_std
    """
    z_score = (xi - xi_mean) / xi_std
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

    #print(f"Z-score: {z_score}, P-value: {p_value}")
    if p_value < 0.05:
        return True
    else:
        return False
    

def evaluate_model_leaspy_on_extreme(longitudinal_model, extreme_values_acc, extreme_dataloader, ids_dataset_extreme, path_to_leaspy_model, device):

    with open(path_to_leaspy_model) as f:
        file = json.load(f)
        xi_mean = file['parameters']['xi_mean']
        xi_std = file['parameters']['xi_std']
    
    print(f"xi_mean: {xi_mean}, xi_std: {xi_std}")
    
    estimator = Leaspy.load(path_to_leaspy_model)

    encodings_df_extreme = build_compatible_leaspy_dataframe(extreme_dataloader, longitudinal_model, device)

    data_extreme = Data.from_dataframe(encodings_df_extreme)
    settings_personalization = AlgorithmSettings('scipy_minimize', use_jacobian=True)
    ip_extreme = estimator.personalize(data_extreme, settings_personalization)
    timepoints = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    acc = 0
    bad_ids = []
    for i, (img, timestep, id) in enumerate(ids_dataset_extreme):

        #print(f"\n\nID: {str(id)}")
        alpha = extreme_values_acc[extreme_values_acc['id'] == id]['alpha'].values[0]
        individual_params = ip_extreme._individual_parameters[str(id)]
        #print(f"Individual parameters: {individual_params}")


        #print(f"Estimated Alpha: {np.exp(individual_params['xi'])}")
        #print(f"Real Alpha: {alpha}")
        bool = statistical_test(individual_params['xi'], xi_mean, xi_std)
        #print(f"Statistical test: {bool}")
        if bool : acc+=1
        else : 
            bad_ids.append((id, alpha, individual_params['xi']))
    #print(acc)
    return acc / len(ids_dataset_extreme), bad_ids