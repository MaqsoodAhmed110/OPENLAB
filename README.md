# Image Classification with CNN Model

This repository contains a Python script (`image_classification.py`) for training a **Convolutional Neural Network (CNN)** model using **TensorFlow** to classify images of strawberry plants into different categories. The script leverages TensorFlow's high-level API, **Keras**, to build and train the model.

---

## Functionality

The Python script performs the following tasks:

1. **Library Imports**:
   - Imports TensorFlow and other essential libraries.

2. **Constant Definitions**:
   - Sets constants such as image size, batch size, channels, and the number of training epochs.

3. **Dataset Loading**:
   - Loads a dataset of strawberry plant images from a specified directory using TensorFlow's `image_dataset_from_directory` function.

4. **Data Visualization**:
   - Visualizes a sample of the dataset using Matplotlib.

5. **Dataset Splitting**:
   - Splits the dataset into training, validation, and test sets.

6. **Data Augmentation**:
   - Applies data augmentation techniques including resizing, rescaling, random flipping, rotation, and zooming using TensorFlow's `tf.keras.layers.experimental.preprocessing` module.

7. **CNN Model Building**:
   - Constructs a CNN model with multiple convolutional and pooling layers followed by fully connected layers.

8. **Model Compilation**:
   - Compiles the model using the Adam optimizer and sparse categorical crossentropy loss.

9. **Model Training**:
   - Trains the model on the training set and evaluates its performance on the validation set.

10. **Performance Visualization**:
    - Displays training and validation accuracy and loss plots to track model progress.

11. **Predictions on Test Data**:
    - Makes predictions on a sample test dataset and visualizes the results.

---

## Dataset

The dataset consists of images of strawberry plants organized into various categories, each representing a unique class label. The directory structure is as follows:

```plaintext
D:\
└───PlantVillage
    └───strawberryplant
        ├───class1
        ├───class2
        └───class3
