# Music Genre Classification Project

This project compares the performance of two neural network models—convolutional neural network models (CNN) and fully connected neural network models (FCNN)—in classifying music genres. The CNN uses spectrogram images as input, while the FCNN uses audio features extracted from audio files.

## Dataset

The GTZAN dataset is deprecated but can be downloaded [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

To use the dataset with this program, first move all spectrogram image files from their respective genre folders to the parent "images_original" folder, then move the entire contents of the downloaded "Data" folder to the local "input" folder.

Run the script: python.exe ./genre_classification_gtzan.py
