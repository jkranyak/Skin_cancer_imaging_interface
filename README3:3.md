# Skin Lesion Classification Project

## Table of Contents
1. [Project Description](#project-description)
2. [Summary](#summary)
3. [Setup Instructions](#setup-instructions)
   - 3.1 [Initial Setup and Library Installation](#initial-setup-and-library-installation)
   - 3.2 [Query the Dataset](#query-the-dataset)
   - 3.3 [Data Preparation and Preprocessing](#data-preparation-and-preprocessing)
   - 3.4 [Splitting Data and Experimentation](#splitting-data-and-experimentation)
   - 3.5 [Model Training](#model-training)
   - 3.6 [Model Summary and Visualization](#model-summary-and-visualization)
   - 3.7 [Further Model Evaluation and Misclassification Analysis](#further-model-evaluation-and-misclassification-analysis)
   - 3.8 [Knowledge Distillation and Fine-tuning](#knowledge-distillation-and-fine-tuning)
   - 3.9 [User Interface Development](#user-interface-development)
4. [Model Details](#model-details)
   - 4.1 [Dual Input Neural Network Architecture](#dual-input-neural-network-architecture)
   - 4.2 [Augmentation and Hyperparameter Optimization](#augmentation-and-hyperparameter-optimization)
   - 4.3 [Callbacks for Model Training](#callbacks-for-model-training)
   - 4.4 [Evaluation Metrics and ROC Curves](#evaluation-metrics-and-roc-curves)
   - 4.5 [Usage and Disclaimer](#usage-and-disclaimer)
5. [Future Directions](#future-directions)
6. [References](#references)

## 1. Project Description
This project focuses on classifying skin lesions using deep learning techniques. The ISIC 2019 dataset, containing images of various skin lesions and associated metadata, is used for model training and testing.

## 2. Summary
The project aims to develop a dual-input model integrating image data and patient metadata to improve diagnostic accuracy. Key aspects include data handling, model architecture design, training strategies, and evaluation methods.

## 3. Setup Instructions
### 3.1 Initial Setup and Library Installation
Install necessary libraries for the project.
### 3.2 Query the Dataset
Interact with the ISIC API to query the dataset, managing data retrieval and filtering based on criteria such as diagnosis and metadata.
### 3.3 Data Preparation and Preprocessing
Handle metadata for the ISIC 2019 dataset, ensuring proper structuring and preparation for analysis.
### 3.4 Splitting Data and Experimentation
Split the dataset into training, validation, and testing sets. Experiment with different model architectures and training strategies.
### 3.5 Model Training
Initialize data generators for training, validation, and testing. Train the dual-input neural network model using TensorFlow.
### 3.6 Model Summary and Visualization
Generate a summary of the model architecture and visualize the model's design. Evaluate predictions and generate a confusion matrix.
### 3.7 Further Model Evaluation and Misclassification Analysis
Evaluate the model's performance on unseen data, analyze misclassifications, and explore methods for improvement.
### 3.8 Knowledge Distillation and Fine-tuning
Explore techniques such as knowledge distillation and fine-tuning to improve model performance.
### 3.9 User Interface Development
Develop a web-based user interface using Gradio to allow users to upload images and receive predictions with explanations.

## 4. Model Details
### 4.1 Dual Input Neural Network Architecture
Detailed breakdown of the model architecture, including image input branch, metadata input branch, and combining branches.
### 4.2 Augmentation and Hyperparameter Optimization
Experimentation with augmentation techniques and hyperparameter optimization using Optuna.
### 4.3 Callbacks for Model Training
Configuration of callbacks such as EarlyStopping and ModelCheckpoint for efficient model training.
### 4.4 Evaluation Metrics and ROC Curves
Evaluation of model performance using metrics such as ROC curves and AUC values.
### 4.5 Usage and Disclaimer
Instructions on how to use the user interface and a disclaimer regarding the limitations of model predictions.

## 5. Future Directions
Potential enhancements and directions for future development, including model expansion and integration with medical systems.

## 6. References
- ISIC 2019 dataset
- TensorFlow documentation
- Gradio documentation
- Optuna documentation
