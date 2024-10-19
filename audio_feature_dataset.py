import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

label_strings = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9,
}

class AudioFeatureDataset(Dataset):
    def __init__(self, audio_feature_path):
        self.df = pd.read_csv(audio_feature_path)
        self.data = self.df.drop(columns=['filename', 'label']).values
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        self.labels = self.df['label'].apply(lambda x: label_strings[x]).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        features = torch.tensor(self.data[i], dtype=torch.float32)
        label = torch.tensor(self.labels[i], dtype=torch.long)
        return features, label

    def train_test_split(self):
        indices = list(range(len(self.df)))
        random.shuffle(indices)
        n = len(self.df)
        n_train = int(0.8 * n)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        train_data = self.df.iloc[train_indices]
        test_data = self.df.iloc[test_indices]
        train_dataset = AudioFeatureDataset.from_dataframe(train_data)
        test_dataset = AudioFeatureDataset.from_dataframe(test_data)
        return train_dataset, test_dataset
    
    @classmethod
    def from_dataframe(cls, dataframe):
        dataset = cls.__new__(cls)
        dataset.df = dataframe
        dataset.data = dataframe.drop(columns=['filename', 'label']).values
        dataset.labels = dataframe['label'].apply(lambda x: label_strings[x]).values
        return dataset