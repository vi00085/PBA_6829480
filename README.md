# Rainfall Prediction and Influence Analysis: A Comparative Study of Machine Learning Techniques on Imbalanced Datasets

This project aims to predict rainfall and analyze the key factors influencing rain using various machine learning techniques on an imbalanced dataset. The dataset consists of 10 years of weather observations in Australia, sourced from Kaggle. The project focuses on handling class imbalance, applying different models, and comparing their performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
  - [Logistic Regression](#logistic-regression)
  - [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
  - [Gradient Boosting Machine (GBM)](#gradient-boosting-machine-gbm)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Naive Bayes](#naive-bayes)
  - [Decision Tree](#decision-tree)
- [Files Included](#files-included)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation and Setup](#installation-and-setup)

## Project Overview

Accurately predicting whether it will rain the next day is critical for a variety of industries, including transportation, energy, and agriculture. The goal of this project is to determine which machine learning model works best with an imbalanced dataset for predicting rainfall, and to identify which factors are the most influential in making these predictions.

The **CRISP-DM** methodology was employed, iterating through stages of data preprocessing, modeling, and evaluation to improve results.

## Dataset

The dataset used in this project consists of the following features:
- **Date**: Date of observation.
- **Location**: Location of the weather station.
- **Weather features**: Temperature, rainfall, evaporation, sunshine, wind speed, pressure, humidity, cloud cover, and more.
- **Target variable**: **RainTomorrow** (boolean) - whether or not it rained the next day.

Data Source: [Kaggle - Weather Dataset Rattle Package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

The dataset contains over **145,000 observations** with various missing values and outliers. Class imbalance is also present, as there are significantly more "No Rain" cases compared to "Rain".

## Models Implemented

### Logistic Regression
- **Purpose**: A baseline model for binary classification.
- **Performance**: Achieved **84% accuracy** with **log transformation** and hyperparameter tuning.
- **Key Findings**: Logistic regression identified **Pressure, Temperature, Windspeed, and Humidity** as the most influential features.

### Artificial Neural Network (ANN)
- **Purpose**: To explore a more complex model with the ability to capture non-linear relationships.
- **Performance**: Achieved high **precision** and favorable recall on the positive class but took longer to train.
- **Best Configuration**: ANN with **SMOTE balancing** for class imbalance and **hyperparameter tuning**.

### Gradient Boosting Machine (GBM)
- **Purpose**: An ensemble method known for handling imbalanced data.
- **Performance**: Achieved **72% accuracy** after hyperparameter tuning.
- **Challenges**: Struggled with class imbalance despite high overall accuracy.

### K-Nearest Neighbors (KNN)
- **Purpose**: A simple instance-based learning algorithm.
- **Performance**: Achieved **82% accuracy**, with the highest **recall** at 96%.
- **Best Configuration**: KNN with **hyperparameter tuning**, **log transformation**, and **feature selection**.

### Naive Bayes
- **Purpose**: A probabilistic classifier for categorical and continuous features.
- **Performance**: Achieved moderate accuracy but excelled in **precision**.
- **Best Configuration**: Naive Bayes on oversampled data with **date** and **location** included.

### Decision Tree
- **Purpose**: A tree-based model for binary classification.
- **Performance**: Achieved **83% accuracy** and was the best model overall in terms of simplicity and interpretability.
- **Key Findings**: Similar to logistic regression, **Pressure, Windspeed, and Humidity** were the most influential factors.

## Files Included

- **`GROUP_12_FINAL_REPORT.pdf`**: The complete project report.
- **`Group_12pba.R`**: R script for the machine learning models and data preprocessing.
- **`Sandeep_6829480.R`**: R script containing the specific contributions from Venkat Sandeep Imandi, including Gradient Boosting and Logistic Regression.
- **`weatherAUS.csv`**: The dataset used for modeling and analysis.

## Results

- **Best Overall Model**: Logistic Regression with log transformation, achieving an accuracy of **84%**, precision of **87%**, and recall of **94%**.
- **Best Recall**: KNN, achieving a recall of **96%**.
- **Best Precision**: Artificial Neural Network (ANN), with high precision on positive class prediction.
- **Feature Importance**: Across multiple models, the most important features for predicting rainfall were **Pressure, Windspeed, Humidity, and Temperature**.

## Conclusion

The project concluded that **Logistic Regression** with log transformation and hyperparameter tuning was the best overall model for predicting rainfall. **KNN** and **ANN** also performed well in terms of recall and precision, respectively. The inclusion of **date** and **location** had minimal impact on most models.

Future improvements could involve collecting more balanced data to improve class prediction for rare events like rain. Logistic regression offers stakeholders a fast and reliable way to predict rainfall for operational planning in industries reliant on weather data.

## Installation and Setup

1. Clone the repository:
   git clone https://github.com/vio0085/PBA_6829480.git

2. Install the required R libraries:
   install.packages(c("caret", "e1071", "pROC", "rpart", "xgboost"))

3. Load the dataset and run the R scripts for each model.
