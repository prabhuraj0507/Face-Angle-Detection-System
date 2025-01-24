# Face-Angle-Detection-System
This project focuses on developing a Convolutional Neural Network (CNN) model to predict the angles of faces in images. The model is trained using Python and TensorFlow, leveraging advanced image preprocessing techniques and data augmentation to achieve high accuracy.

# Project Overview
Objective: To build a CNN model that accurately predicts the x and y angles of faces in images.
Dataset: A collection of 10k images labeled with face angles.
Model: A CNN model with multiple convolutional layers, max-pooling layers, and dense layers.
Evaluation Metrics: Mean Absolute Error (MAE) and Mean Squared Error (MSE).
# Features
Advanced Preprocessing: Includes Sobel edge detection to highlight important features in images.
Data Augmentation: Uses techniques like random flipping, brightness adjustments, and more to enhance model robustness.
Learning Rate Scheduling: Adjusts the learning rate dynamically during training for optimal performance.
Visualization: Provides visual comparisons of true vs. predicted values and training history plots.
# Model Architecture
Input Layer: 64x64 grayscale images
Convolutional Layers:
Conv2D: 64 filters, 3x3 kernel
Conv2D: 128 filters, 3x3 kernel
Conv2D: 256 filters, 3x3 kernel
MaxPooling Layers: 2x2 pool size
Fully Connected Layers:
Dense: 256 neurons, ReLU activation
Dropout: 50%
Output: 2 neurons for x and y angles
# Installation
Clone the repository:

git clone https://github.com/yourusername/face-angle-detection.git
cd face-angle-detection
Install the required packages: already present in code

Set up the dataset:

Place your image dataset in the directory D:\\03_Projects\\01_CNN\\DATA1.
Data Preprocessing
The images are preprocessed using Sobel edge detection to enhance feature extraction. Labels are extracted from filenames and converted to numerical values.

Data Augmentation
Random flipping, brightness adjustments, and other augmentation techniques are applied to the training data to improve model generalization.

Model Training
The model is trained using a learning rate scheduler to dynamically adjust the learning rate, improving convergence and performance.

# Acknowledgments
TensorFlow for providing the framework to build and train the model.
OpenCV for image processing functions.
