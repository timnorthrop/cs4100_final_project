import matplotlib.pyplot as plt

import torch
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from genre_classifier_fcnn import GenreClassifierFCNN
from genre_classifier_cnn import GenreClassifierCNN
from spectrogram_dataset import SpectrogramDataset
from audio_feature_dataset import AudioFeatureDataset

transform = transforms.Compose([transforms.Resize((216, 144)),
                                transforms.ToImage(),
                                transforms.ToDtype(torch.float32, scale=True),])

cnn_dataset = SpectrogramDataset('input/images_original/', transform=transform)
cnn_train_dataset, cnn_test_dataset = cnn_dataset.train_test_split()

fcnn_dataset = AudioFeatureDataset('input/features_30_sec.csv')
fcnn_train_dataset, fcnn_test_dataset = fcnn_dataset.train_test_split()

cnn_trainloader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True)
cnn_testloader = DataLoader(cnn_test_dataset, batch_size=32, shuffle=False)

fcnn_trainloader = DataLoader(fcnn_train_dataset, batch_size=32, shuffle=True)
fcnn_testloader = DataLoader(fcnn_test_dataset, batch_size=32, shuffle=False)

cnn_model = GenreClassifierCNN()
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
cnn_loss_arr = []

fcnn_model = GenreClassifierFCNN()
fcnn_criterion = nn.CrossEntropyLoss()
fcnn_optimizer = optim.Adam(fcnn_model.parameters(), lr=0.001)
fcnn_loss_arr = []

# taken from assignment 3
for epoch in range(30):
    cnn_running_loss = 0.0
    for i, data in enumerate(cnn_trainloader, 0):
        inputs, labels = data
        cnn_optimizer.zero_grad()
        outputs = cnn_model(inputs)
        cnn_loss = cnn_criterion(outputs, labels)
        cnn_loss.backward()
        cnn_optimizer.step()
        cnn_running_loss += cnn_loss.item()

    fcnn_running_loss = 0.0
    for i, data in enumerate(fcnn_trainloader, 0):
        inputs, labels = data
        fcnn_optimizer.zero_grad()
        outputs = fcnn_model(inputs)
        fcnn_loss = fcnn_criterion(outputs, labels)
        fcnn_loss.backward()
        fcnn_optimizer.step()
        fcnn_running_loss += fcnn_loss.item()
    
    cnn_loss_arr.append(cnn_running_loss)
    fcnn_loss_arr.append(fcnn_running_loss)
    print(f'CNN epoch {epoch+1} loss: {cnn_loss.item()}')
    print(f'FCNN epoch {epoch+1} loss: {fcnn_loss.item()}')

cnn_correct = 0
fcnn_correct = 0
total = 0
cnn_model.eval()
fcnn_model.eval()
with torch.no_grad():
    for data in cnn_testloader:
        images, labels = data
        predictions = cnn_model(images)
        _, predicted = torch.max(predictions, 1)
        cnn_correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    for data in fcnn_testloader:
        features, labels = data
        predictions = fcnn_model(features)
        _, predicted = torch.max(predictions, 1)
        fcnn_correct += (predicted == labels).sum().item()

print('CNN accuracy: ', cnn_correct/total)
print('FCNN accuracy: ', fcnn_correct/total)