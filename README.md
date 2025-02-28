Creditworthiness Prediction Model

This repository contains a machine learning project that predicts a customer’s creditworthiness based on financial behavior, transaction history, and demographic features. The model helps financial institutions assess the risk associated with customer lending and make more informed credit decisions.

Project Overview

In this project, we develop a classification model that predicts whether a customer is creditworthy or not, using features such as loan types, missed payments, demographic information, and other financial behavior indicators. The project involves the following key steps:

- Data Preprocessing: Handling missing values, encoding categorical variables, and normalizing data.
- Feature Engineering: Creating new features from transaction history and aggregating key financial metrics.
- Model Building: Training a machine learning model using Random Forest and other algorithms to classify customers as creditworthy or not.
- Model Evaluation: Assessing the model’s performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- Hyperparameter Tuning: Using GridSearchCV to optimize the model for better performance.
- Visualization: Exploratory data analysis and visualizations to understand the trends and relationships between variables.

Features:

- Data Preprocessing: Includes handling missing values and encoding categorical variables such as marital status and education level.
- Imbalanced Data Handling: Uses SMOTE to address class imbalance in the creditworthiness dataset.
- Model Training: Implements Random Forest classifier, with hyperparameter tuning using GridSearchCV.
- Evaluation Metrics: Provides detailed performance metrics like confusion matrix, ROC-AUC, and feature importance.
- Exploratory Data Analysis (EDA): Visualizes data distributions, correlations, and relationships between variables and creditworthiness.

Tech Stack:

- Programming Language: Python

  Libraries:

- Data Manipulation: Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn, RandomForestClassifier, SMOTE
- Hyperparameter Tuning: GridSearchCV
- Model Evaluation: Confusion Matrix, ROC-AUC
