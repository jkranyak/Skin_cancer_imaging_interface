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
5. [CancerNet-SCa: Deep Neural Network for Skin Cancer Detection](#CancerNet-SCa: Deep Neural Network for Skin Cancer Detection)
6. [Future Directions](#future-directions)
7. [References](#references)


<a name="project-description"></a>
## 1. Project Description
This project focuses on classifying skin lesions using deep learning techniques. The ISIC 2019 dataset, containing images of various skin lesions and associated metadata, is used for model training and testing.


<a name="summary"></a>
## 2. Summary
The project aims to develop a dual-input model integrating image data and patient metadata to improve diagnostic accuracy. Key aspects include data handling, model architecture design, training strategies, and evaluation methods.


<a name="setup-instructions"></a>
## 3. Setup Instructions
### 3.1 Initial Setup and Library Installation
Install necessary libraries for the project.
### 3.2 Query the Dataset 
Interact with the ISIC API to query the dataset, managing data retrieval and filtering based on criteria such as diagnosis and metadata.
(provided cleaned data is available at full_metadata.csv this df is code ready, the objects are called to .drop, but are needed to locate images in workflow)
### 3.3 Data Preparation and Preprocessing
Handle metadata for the ISIC 2019 dataset, ensuring proper structuring and preparation for analysis.
### 3.4 Splitting Data and Experimentation
Split the dataset into training, validation, and testing sets. Experiment with different model architectures and training strategies.
(See file model_diagram for final classifier model structure, a multi input neural network)
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
(model represented in code at master_project3.ipynb & master_project_model_1.ipynb in different phases)
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


# 5. CancerNet-SCa: Deep Neural Network for Skin Cancer Detection
(refer to Main_project3_model_2.ipynb in files for model)

CancerNet-SCa is a groundbreaking deep neural network architecture meticulously crafted for the explicit purpose of detecting skin cancer from dermoscopy images. Developed as part of the Cancer-Net initiative, this model represents a significant leap forward in the realm of computer-aided diagnosis for dermatological conditions. Below is a detailed overview of its key features and functionalities:

## Motivation and Background:
- Skin cancer stands as the most prevalent form of cancer diagnosed in the U.S., not only posing substantial threats to health and well-being but also incurring significant economic burdens due to treatment costs.
- Early detection plays a pivotal role in the effective treatment and management of skin cancer, with promising prognoses associated with timely intervention, particularly through methodologies such as dermoscopy examination.

## Inspiration and Development:
- Inspired by the remarkable advancements in deep learning methodologies and spurred by the ethos of open-source collaboration prevalent in the research community, [CancerNet-SCa](https://arxiv.org/abs/2011.10702) was conceived and developed.
- It represents a suite of deep neural network designs meticulously tailored to tackle the challenge of skin cancer detection from dermoscopy images.

## Design and Features:
- CancerNet-SCa marks the first instance of machine-designed deep neural network architectures explicitly optimized for skin cancer detection.
- Among its innovative design elements is the incorporation of attention condensers, facilitating an efficient self-attention mechanism aimed at enhancing detection accuracy.
- These designs exhibit superior performance metrics when compared to established architectures like ResNet-50, achieving commendable accuracy while simultaneously reducing both architectural complexity and computational overhead.

## Performance and Decision Making:
- Extensive experimentation and evaluation utilizing datasets such as the International Skin Imaging Collaboration (ISIC) dataset underscore CancerNet-SCa's robust performance in skin cancer detection.
- Notably, the model demonstrates an ability to discern diagnostically relevant critical factors, eschewing irrelevant visual indicators and imaging artifacts, thereby bolstering diagnostic accuracy.

## Open Source and Encouragement:
- CancerNet-SCa represents a significant contribution to the open-source landscape, being made readily available to researchers, clinicians, and citizen data scientists.
- While not positioned as a production-ready screening solution, its release in [open-source, open-access](https://arxiv.org/abs/2011.10702) format is intended to catalyze further advancements in skin cancer detection methodologies through collaborative exploration and refinement.

## Model Architecture:
- CancerNet-SCa embodies a multi-input, binary classification model capable of assimilating both image and metadata features.
- Its architecture comprises convolutional and pooling layers for image processing, alongside dense layers for metadata analysis, with the two branches concatenated for subsequent processing.
- The model leverages a pre-trained convolutional neural network (CNN) for image input, possibly utilizing architectures like VGG16 or ResNet50.

## Binary Classifier and Tuning:
- Engineered for binary classification, CancerNet-SCa employs a sigmoid activation function in its output layer, with binary cross-entropy loss function and accuracy as evaluation metrics.
- To address data imbalances inherent in skin cancer datasets, the model utilizes class weights to ensure equitable learning.

## Additional Features:
- CancerNet-SCa incorporates various optimization strategies, including data augmentation for image inputs and callbacks such as early stopping and model checkpointing during training.
- Its training regimen encompasses a dataset rich in both image and metadata features, including patient-specific information such as age and sex, augmenting the model's diagnostic capabilities.
- See files in experimental_models folder for GAN attempt at supplementing images for models, and a code attempt at using a conglomeration of modern methds extracted from research papers on medical imaging and engineered through research and code chat bots. Both were not succesful.
  
In summation, [CancerNet-SCa](https://arxiv.org/abs/2011.10702) stands as a testament to the transformative potential of deep learning methodologies in the domain of medical imaging analysis. Its development heralds a new era of precision diagnostics for dermatological conditions, underpinned by the principles of collaboration, transparency, and accessibility.

## 5. Future Directions
Potential enhancements and directions for future development, including model expansion and integration with medical systems.

## 6. References
- ISIC 2019 dataset
- TensorFlow documentation
- Gradio documentation
- Optuna documentation
