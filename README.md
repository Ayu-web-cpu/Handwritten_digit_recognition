# Digit Recognition CNN

This project implements a Convolutional Neural Network (CNN) for digit recognition using the MNIST dataset. The model is built using PyTorch and includes training and testing functionalities.

## Project Structure

```
digit-recognition-cnn
├── src
│   ├── model.py        # Defines the CNN architecture
│   ├── train.py        # Contains the training loop
│   ├── test.py         # Evaluates the trained model
│   ├── dataset.py      # Handles data loading and preprocessing
├── requirements.txt     # Lists project dependencies
└── README.md            # Project documentation
'''
```
## Model Architecture

The CNN model is defined in `src/model.py` and consists of several convolutional layers followed by fully connected layers. The architecture is designed to effectively learn features from the MNIST dataset.

## Training Process

The training loop is implemented in `src/train.py`, where the model is trained using the training dataset, and the loss is monitored. The trained model can be saved to disk for later use.

# Evaluation

The evaluation of the model is performed in `src/test.py`, where the accuracy and loss on the test dataset are computed to assess the model's performance.

# How it works
Imports:
Uses PyTorch (torch), torchvision’s datasets and transforms, and PyTorch’s DataLoader.
Transform:
Converts images to tensors and normalizes them to have mean 0.5 and std 0.5.

Datasets:
Downloads and loads the MNIST training and test datasets, applying the transform.

DataLoaders:
Wraps the datasets in DataLoader objects for easy batch loading during training and testing.

train_loader shuffles the data for training.
test_loader does not shuffle.
Return:
Returns both train_loader and test_loader for use in your training and evaluation scripts.



