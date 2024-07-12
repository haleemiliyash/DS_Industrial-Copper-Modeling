# DS_Industrial-Copper-Modeling

## Introduction of project:
Project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling price and Status classification. Manual predictions can be time-consuming and may not result in optimal pricing decisions or accurately capture status. The models will utilize advanced techniques such as data standardization, outlier detection and handling, handling data in the wrong format, identifying the distribution of data, and leveraging tree-based models, specifically the decision tree algorithm, to predict the selling price and leads accurately.

## Regression model:
The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data standardization, outlier detection and handling, handling data in wrong format, identifying the distribution of data, and leveraging tree-based models, specifically the RandomForeast regressor algorithm.

## Classification model:
Another area where the copper industry faces challenges is in capturing the status. A Status classification model is a system for evaluating and classifying status based on how likely they are to become a customer. You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.

## Overview of project:
Explore the dataset and format of data.
Transforming the data into a suitable format and performing any necessary cleaning and pre-processing steps.
Exploring skewness and outliers in the dataset.
Developed a machine learning regression model which predicts the continuous variable 'Selling_Price' using the Random Forest regressor.
Developed a machine learning classification model which predicts the Status: WON or LOST using the RandomForest Classifier.
Created a Streamlit page where you can insert each column value and get the Selling_Price predicted value or Status (Won/Lost).

## Libraries Used:
NumPy
Pandas
Scikit-learn
json
Pickel
Matplotlib
