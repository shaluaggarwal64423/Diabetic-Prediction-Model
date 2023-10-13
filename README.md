# Diabetic-Prediction-Model
This repository contains code for a diabetes prediction model that uses machine learning techniques to predict whether an individual is diabetic or not based on various health metrics. The model utilizes the Diabetes dataset and offers insights into data preprocessing, model building, evaluation, and results analysis.

## Problem Understanding
The primary objective of this project is to develop a predictive model for diabetes based on health-related attributes. The problem involves creating a machine learning model that can accurately classify individuals into two categories: diabetic or non-diabetic, using features like Glucose level, Blood Pressure, BMI, and more.

## Approach
The code follows a systematic approach to build and evaluate the diabetes prediction model:

### 1.	Data Preprocessing:
•	Load the dataset from a CSV file.
•	Handle missing values by replacing them with appropriate statistics (mean or median).
•	Detect and remove outliers for data integrity.

### 2.	Exploratory Data Analysis (EDA):
•	Visualize data distributions using histograms.
•	Generate pair plots and heatmaps to analyze feature correlations before and after data cleaning.

### 3.	Model Building:
•	Split the dataset into training and testing sets.
•	Normalize the features using Standard Scaler.
•	Build several machine learning models, including:
•	Logistic Regression
•	Decision Tree Classifier
•	Random Forest Classifier
•	K-Nearest Neighbors
•	Support Vector Machine (SVM)
•	XGBoost Classifier
•	Artificial Neural Network (ANN)

### 4.	Model Evaluation:
•	Calculate various metrics, such as accuracy, F1-score, Mean Squared Error (MSE), R2 Score, Precision, Recall, and Confusion Matrix for each model.

### 5.	Results Analysis:
Based on the evaluation metrics for various machine learning models applied to the diabetes-related dataset, several observations can be made:
•	 Logistic Regression and SVM achieved the highest accuracy (approximately 76.77%), although their F1-scores and precision for class 1 were relatively lower compared to other models.
•	Decision Tree and Random Forest models performed reasonably well, achieving accuracy scores of approximately 70.87% and 75.20%, respectively.
•	K-Nearest Neighbors exhibited the lowest accuracy among the models, with an accuracy score of approximately 67.72%.
•	The artificial neural network (ANN) model had the lowest F1-score and precision for class 1, indicating room for improvement in its performance.

### Additional Notes
•	The dataset used for diabetes prediction should be placed in the same directory as the code file.
•	Modify file paths as needed if you store the dataset in a different location.


