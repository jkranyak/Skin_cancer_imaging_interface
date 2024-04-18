# Skin Lesion Classification Project

## Overview

This project focuses on classifying skin lesions using deep learning techniques. The dataset used for training and testing the model is the ISIC 2019 dataset, which consists of images of various skin lesions along with associated metadata.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Additional Information](#additional-information)

## Overview

This project aims to classify skin lesions into different categories using deep learning models. It utilizes the ISIC 2019 dataset, which contains images of skin lesions along with metadata such as age, sex, and diagnosis.

## Requirements

To run this project, ensure you have the following dependencies installed:

- Google Colab: For running the Python code in a Jupyter notebook environment.
- TensorFlow: Deep learning framework for building and training neural networks.
- ISIC CLI: Command-line interface for accessing the ISIC dataset.
- Kaggle API: Allows downloading datasets from Kaggle.
- scikit-learn: Library for machine learning algorithms and tools.

## Setup

1. **Mount Google Drive:** Mount your Google Drive to access and store files. Run the following code in a Google Colab notebook:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Install Dependencies:** Install the required Python packages using pip:

    ```bash
    !pip install tensorflow isic-cli kaggle scikit-learn tf-keras-vis
    ```

3. **Configure Kaggle API:** If you haven't already, you need to configure the Kaggle API to download datasets. Ensure you have a `kaggle.json` file with your API credentials and upload it to your Google Drive. Then run the following code to move it to the appropriate directory:

    ```python
    import os

    current_path = '/content/kaggle.json'
    desired_path = '/root/.kaggle/kaggle.json'

    if os.path.exists(current_path):
        os.makedirs(os.path.dirname(desired_path), exist_ok=True)
        os.rename(current_path, desired_path)
        os.chmod(desired_path, 0o600)
    else:
        print(f"Error: '{current_path}' does not exist. Please upload the file.")
    ```

4. **Login to ISIC:** Use the ISIC CLI to log in and access the dataset:

    ```bash
    !isic user login
    ```

5. **Download Dataset:** Download the ISIC 2019 dataset using the Kaggle API:

    ```bash
    !kaggle datasets download -d andrewmvd/isic-2019
    ```

6. **Extract Dataset:** Unzip the downloaded dataset:

    ```bash
    !unzip -q isic-2019.zip
    ```

## Data Preparation

1. **Load Metadata:** Load the metadata associated with the ISIC dataset:

    ```python
    import pandas as pd

    metadata = pd.read_csv('/content/ISIC_2019_Training_Metadata.csv')
    ```

2. **Merge Metadata:** Merge the metadata with ground truth data if necessary:

    ```python
    ground_truth = pd.read_csv('/content/ISIC_2019_Training_GroundTruth.csv')
    full_metadata = pd.merge(ground_truth, metadata, on='image', how='left')
    ```

3. **Correct Image Paths:** Correct the base path in the image paths column:

    ```python
    correct_base_path = "/content/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
    full_metadata['image_path'] = full_metadata['image'].apply(lambda x: f"{correct_base_path}/{x}.jpg")
    ```

4. **Split Data:** Split the dataset into training, validation, and test sets:

    ```python
    from sklearn.model_selection import train_test_split

    train_val_data, test_data = train_test_split(full_metadata, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)
    ```

## Model Training and Evaluation

### Model Architecture

The skin lesion classification model employs a convolutional neural network (CNN) architecture, specifically tailored for image classification tasks. The architecture typically consists of convolutional layers followed by pooling layers to extract hierarchical features from input images. Batch normalization and dropout layers are incorporated to improve model generalization and prevent overfitting. The final layers include fully connected layers with softmax activation for multi-class classification.

### Training Procedure

1. **Data Augmentation:** Augment the training data using techniques such as rotation, shifting, zooming, and flipping. Data augmentation helps increase the diversity of training samples and improves model robustness.
   
2. **Class Balancing:** Address class imbalance by applying techniques such as oversampling or undersampling. Class balancing ensures that the model is trained effectively on all classes, preventing bias towards majority classes.

3. **Compile Model:** Compile the CNN model using appropriate loss functions (e.g., categorical cross-entropy), optimizers (e.g., Adam), and evaluation metrics (e.g., accuracy). Adjust hyperparameters such as learning rate based on model performance on the validation set.

4. **Train Model:** Train the compiled model on the training data using batch training. Monitor training metrics such as loss and accuracy to assess model convergence and performance.

5. **Validate Model:** Evaluate the trained model on the validation set at regular intervals during training. Use early stopping based on validation loss or accuracy to prevent overfitting and determine the optimal number of training epochs.

6. **Fine-Tuning:** Optionally, perform fine-tuning by unfreezing selected layers of the pre-trained CNN model and retraining them with a smaller learning rate. Fine-tuning helps adapt the model to the specific characteristics of the skin lesion dataset.

### Evaluation Metrics

Evaluate the trained model on the test set using various evaluation metrics to assess its performance:

- **Accuracy:** The proportion of correctly classified samples out of the total number of samples in the test set.
  
- **Precision:** The proportion of true positive predictions (correctly classified positive samples) out of the total positive predictions.

- **Recall (Sensitivity):** The proportion of true positive predictions out of the total actual positive samples in the test set.

- **F1 Score:** The harmonic mean of precision and recall, providing a balanced measure of model performance.

- **Confusion Matrix:** A matrix that summarizes the actual and predicted classifications, enabling analysis of model errors and misclassifications across different classes.

### Interpretation and Visualization

Interpret the model predictions and visualize its performance using techniques such as Grad-CAM (Gradient-weighted Class Activation Mapping). Grad-CAM generates heatmaps highlighting regions of the input image that contribute most to the model's prediction, aiding in model interpretation and debugging.

### Deployment and Continuous Monitoring

Deploy the trained model in a production environment for real-world applications such as dermatology clinics or telemedicine platforms. Implement continuous monitoring to track model performance, detect drift, and ensure consistent and reliable performance over time.

---

This section provides an in-depth overview of the model training and evaluation process, covering model architecture, training procedures, evaluation metrics, interpretation techniques, and deployment considerations. It serves as a comprehensive guide for understanding and implementing the skin lesion classification model.


## Additional Information

- For more information about the ISIC dataset and skin lesion classification, you can refer to the [official ISIC website](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery).


## Directory Structure
