# Skin Lesion Classification Project

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

- Train and evaluate your model using the provided metadata and image data. You can use deep learning frameworks like TensorFlow to build your classification model.

## Additional Information

- For more information about the ISIC dataset and skin lesion classification, you can refer to the [official ISIC website](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery).
