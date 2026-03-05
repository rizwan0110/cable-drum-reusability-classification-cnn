# Overview

This project focuses on using deep learning to classify cable drums as reusable or non-reusable based on their physical condition. The objective was to help a recycling company automate visual inspection and improve sustainability by identifying drums that can be reused.

The project was completed as part my Deep Learning course in the AI Engineering Master's program at Jönköping University.

The study compares multiple deep learning architectures and data augmentation techniques to determine the most effective approach for image classification. 

# Business Problem

Cable drums are widely used for transporting cables. Over time, they may become damaged and unsuitable for reuse.

Traditionally, companies rely on manual visual inspection, which is:

- Time-consuming
- Labor-intensive
- Prone to human error

The goal of this project was to develop an automated image classification system that can detect reusable cable drums and support sustainable recycling operations.

The dataset was provided by Drumster, a cable drum recycling company based in Sweden. 

# Data

The dataset consisted of images of cable drums in two categories:

1. Non-Reusable Drums

- Broken drums
- Non-standard drums

A total of 12,469 images

2. Reusable Drums

762 images of Drums meeting company recycling standards

A significant class imbalance existed between reusable and non-reusable images, which required augmentation techniques during training. 


# Methodology

The project followed a deep learning workflow:

- Data Cleaning and Image Validation
- Handling Class Imbalance
- Data Augmentation
- Model Training
- Model Evaluation

# Model Explainability
To improve model generalization, two augmentation techniques were applied:
- Geometric transformations (rotation, flipping, zoom)
- Color jittering (brightness, contrast, saturation, hue)
- Images were resized to 224×224 pixels and normalized before training. 

# Models Evaluated

Two pre-trained CNN architectures were used:

- VGG16
- ResNet50

Both models were trained using transfer learning with ImageNet weights and evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

Grad-CAM visualization was also used to explain the model’s predictions.

# Results

The best performing model was:
VGG16 + Color Jittering Data Augmentation

Key performance metrics:

- Validation Accuracy: ~88%
- F1 Score: ~92%


The results showed that color augmentation significantly improved model performance compared to geometric transformations. 

Grad-CAM visualizations confirmed that the model focused on damaged regions of the drum when identifying non-reusable drums. 

# Tools & Technologies
- Python
- TensorFlow / Keras
- NumPy
- Image Data Augmentation
- Convolutional Neural Networks (CNN)
- Grad-CAM Visualization

_The dataset and the code cannot be shared due to company confidentiality._
