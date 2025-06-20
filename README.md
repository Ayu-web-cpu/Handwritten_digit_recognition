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
│   └── utils.py        # Utility functions for the project
├── requirements.txt     # Lists project dependencies
└── README.md            # Project documentation
'''
## Model Architecture

The CNN model is defined in `src/model.py` and consists of several convolutional layers followed by fully connected layers. The architecture is designed to effectively learn features from the MNIST dataset.

## Training Process

The training loop is implemented in `src/train.py`, where the model is trained using the training dataset, and the loss is monitored. The trained model can be saved to disk for later use.

## Evaluation

The evaluation of the model is performed in `src/test.py`, where the accuracy and loss on the test dataset are computed to assess the model's performance.

