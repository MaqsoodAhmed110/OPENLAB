Image classification with CNN model
This repository contains a Python script (image_classification.py) for training a convolutional neural network (CNN) model using TensorFlow to classify images of strawberry plants into different categories.
The script utilizes TensorFlow's high-level API, Keras, for building and training the model.
Functionality
The Python code performs the following tasks:
Imports TensorFlow and necessary libraries.
Defines constants such as image size, batch size, channels, and number of epochs.
Loads the dataset of strawberry plant images from the specified directory using TensorFlow's image_dataset_from_directory function.
Visualizes a sample of the dataset using matplotlib.
Splits the dataset into training, validation, and test sets.
Implements data augmentation techniques including resizing, rescaling, random flipping, rotation, and zooming using TensorFlow's experimental.preprocessing module.
Builds a CNN model using TensorFlow's Keras API with convolutional and pooling layers, followed by fully connected layers.
Compiles the model with an Adam optimizer and sparse categorical crossentropy loss.
Trains the model on the training dataset and evaluates its performance on the validation dataset.
Displays training and validation accuracy and loss plots.
Makes predictions on a sample test dataset and displays the results.
Dataset
The dataset used for training and testing consists of images of strawberry plants. It is organized into different categories, each representing a class label. 
The dataset is structured in the following directory format:
D:\
│
└───PlantVillage
    └───strawberryplant
        ├───class1
        ├───class2
        └───class3
Results
The model's performance is evaluated on a separate test dataset, and classification accuracy and loss metrics are reported.
