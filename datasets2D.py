import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset




class Dataset2D(Dataset):
    """
    Dataset made to train the autoencoder without the longitudinal component on the synthetic starmen dataset.
    Returns only the images from the input dataframe (format img .npy).
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float32))
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        summary_rows = self.dataframe.iloc[idx]
        img_path = summary_rows['path']
        image = np.load(img_path)
        image = torch.from_numpy(image).float()
        return image.unsqueeze(0)
    

class DatasetLongitudinal2D(Dataset):
    """
    Dataset made to train the autoencoder with the longitudinal component on the synthetic starmen dataset.
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float32))
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        summary_row = self.dataframe.iloc[idx]
        img_path = summary_row['path']
        image = np.load(img_path)
        image = torch.from_numpy(image).float()
        id = summary_row['id']
        timestep = summary_row['age']
        return image.unsqueeze(0), id, timestep
    


class ID_Dataset2D(Dataset):
    """
    Dataset made to return timesteps, and images from id.
    """


    def __init__(self, dataframe):
        self.dataframe =    dataframe.sort_values(['id', 'timestep'])
        self.list_patient_ids = self.dataframe['id'].unique().tolist()

    def __len__(self):
        return len(self.list_patient_ids)

    def __getitem__(self, idx):
        """
        returns observations for an individual, the time of observation,the id of the individual
        images.shape = number_of_observations x 1 x Depth x Height x Width
        """

        patient_id = self.list_patient_ids[idx]
        summary_rows = self.dataframe[self.dataframe['id'] == patient_id].sort_values(['id', 'timestep'])
        timesteps = summary_rows['timestep'].tolist()
        images = [torch.from_numpy(np.load(summary_rows.iloc[i]['path'])).float() for i in range(len(summary_rows))]
        images = torch.stack(images)
    
        return images, timesteps, patient_id