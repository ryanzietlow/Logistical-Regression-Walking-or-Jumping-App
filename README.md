# Logistical-Regression-Walking-or-Jumping-App

Here's a detailed GitHub README for your project:

---

# Logistical Regression Walking or Jumping App

This project aims to create a desktop application that can distinguish between the actions of walking and jumping using accelerometer data. The application takes data from CSV files, processes it, and outputs a new CSV file labeling each action as either walking or jumping.

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Storing](#data-storing)
- [Visualization](#visualization)
- [Pre-Processing](#pre-processing)
- [Feature Extraction and Normalization](#feature-extraction-and-normalization)
- [Training the Classifier](#training-the-classifier)
- [Model Deployment](#model-deployment)
- [Participation](#participation)
- [Bibliography](#bibliography)

## Introduction

The goal of this project is to construct a desktop application that distinguishes between walking and jumping actions. The application processes accelerometer data along the X, Y, and Z axes from a CSV file and outputs a new CSV file that labels each action as walking or jumping.

## Data Collection

Data was collected using a mobile application called Phyphox, which utilizes built-in sensors such as accelerometers, gyroscopes, and magnetometers. Each team member collected three minutes of both walking and jumping data with the phone in three separate positions: right hand, left pocket, and inside the back of their waistband. The data was then exported into CSV files and labeled with 0 for walking and 1 for jumping.

## Data Storing

To correctly store the raw data, several steps were taken:
1. **Labeling the Data**: A function was created to label the raw data.
2. **Shuffling the Data**: A shuffle function was created to randomize the data to avoid any time-based patterns.
3. **Combining the Data**: Data from all three participants were merged into single files for walking and jumping.
4. **Splitting the Data**: The dataset was split into 90% training and 10% testing data.
5. **Loading into HDF5 Files**: The labeled, shuffled, combined, and split data were loaded into HDF5 files.

## Visualization

The collected data was visualized using 3D scatter plots and time-series graphs to observe the acceleration data in the X, Y, and Z directions. This visualization helped in understanding the distinct patterns in walking and jumping data.

![Walking Acceleration](images/walking_acceleration.png)
![Jumping Acceleration](images/jumping_acceleration.png)

## Pre-Processing

To reduce noise and ensure the accuracy of the classifier, a rolling mean with a window size of 5 was applied to the data. This smoothed the data and reduced rapid fluctuations, making trends more evident for the model.

## Feature Extraction and Normalization

Features such as mean, max, min, median, standard deviation, skew, kurtosis, variance, and sum were extracted for the X, Y, and Z accelerations. These features were then normalized using the StandardScaler from the sklearn library to ensure each feature contributes equally to the model's performance.

## Training the Classifier

The classifier was trained using a machine learning pipeline with StandardScaler and LogisticRegression. The model was evaluated using accuracy, recall, confusion matrix, ROC curve, and AUC to ensure its performance.

![Confusion Matrix](images/confusion_matrix.png)
![ROC Curve](images/roc_curve.png)
![Model Accuracy](images/model_accuracy.png)

## Model Deployment

The model was deployed into a graphical user interface (GUI) using Tkinter. The GUI allows users to input a CSV file, processes the data, and outputs a labeled CSV file along with a visualization of the results. However, there was an issue with feature names not matching between the training model and the prediction phase, which the team was unable to resolve.

![GUI](images/gui.png)
![Error](images/error.png)
