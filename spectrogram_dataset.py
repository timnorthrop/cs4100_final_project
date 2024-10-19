import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random

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

class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, transform=None):
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.filenames = os.listdir(spectrogram_dir)
        self.data = []

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        img_name = os.path.join(self.spectrogram_dir, self.filenames[i])
        image = Image.open(img_name).convert('RGB')
        label_string = self.filenames[i].split('0', maxsplit=1)[0]
        label = label_strings[label_string]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

    def train_test_split(self):
        n = len(self.filenames)
        n_train = int(0.8 * n)
        random.shuffle(self.filenames)
        train_filenames = self.filenames[:n_train]
        test_filenames = self.filenames[n_train:]
        train_dataset = SpectrogramDataset(self.spectrogram_dir, transform=self.transform)
        train_dataset.filenames = train_filenames
        test_dataset = SpectrogramDataset(self.spectrogram_dir, transform=self.transform)
        test_dataset.filenames = test_filenames
        return train_dataset, test_dataset
