# Handwritten Digit Recognition (MNIST) with PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch to recognize handwritten digits from the MNIST dataset. It includes model definition, training, evaluation, and a simple UI for drawing and predicting digits.

## Problem Statement

Handwritten text varies widely, making automated digit recognition a challenging task. This project demonstrates how AI, specifically CNNs, can help read such inputs—useful in educational technology and smart exam-checking tools.

## Objective

- Build a CNN-based digit recognizer using the MNIST dataset.
- Achieve over 98% accuracy on test data.
- Provide a simple UI for users to draw digits and get instant predictions.

## Project Structure

```
Handwritten_digit_recognition/
├── src/
│   ├── model.py      # CNN model definition
│   ├── train.py      # Training script
│   ├── test.py       # Testing/evaluation script
│   ├── dataset.py    # Data loading and preprocessing
│   ├── ui.py         # Tkinter-based UI for digit drawing and prediction
├── requirements.txt  # List of dependencies
└── README.md         # Project documentation
```

## Requirements

- Python 3.x
- torch, torchvision
- pillow
- numpy
- tk
- (Optional: streamlit and streamlit-drawable-canvas if you want to experiment with Streamlit UI)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python src/train.py
```
- Downloads MNIST, trains the CNN, and saves as `digit_cnn.pth`.

### 2. Test the Model

```bash
python src/test.py
```
- Loads the trained model and prints test accuracy.

### 3. Run the Digit Drawing UI

```bash
python src/ui.py
```
- Opens a Tkinter window. Draw a digit and click "Predict" to see the result.

## Model Architecture

Implemented in `src/model.py`:
- 2 convolutional layers with ReLU activation and max pooling
- Dropout for regularization
- Fully connected layers, outputting 10 digit classes (0-9)

## File Overview

- `src/model.py`: Defines the `DigitCNN` class, a PyTorch CNN for digit recognition.
- `src/train.py`: Trains the CNN using the MNIST dataset, reports accuracy, and saves the model.
- `src/test.py`: Loads the saved model and computes test accuracy.
- `src/ui.py`: Tkinter-based GUI for drawing digits and predicting with the trained model.
- `src/dataset.py`: Utility for loading and preprocessing MNIST data.
- `requirements.txt`: Python dependencies.

## Example Results

- Training and evaluation scripts will print loss and accuracy per epoch.
- Typical test accuracy: **over 98%**.

## References

- [PyTorch](https://pytorch.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## License

MIT License

---







