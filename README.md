# Container Water Prediction using LSTM
This project demonstrates how to use a Transformer-based model for time series prediction, implemented in PyTorch. The model is trained and evaluated on a dataset that has been preprocessed and normalized.

## Overview
The main objective of this project is to predict future values of a time series using a Transformer model. The Transformer architecture is employed due to its strong performance in sequence-based problems, including time series forecasting.

## Key Components:
Data Preprocessing: The dataset is split into training and testing sets, followed by normalization using MinMaxScaler.
Model: A custom Transformer model is created with multi-head self-attention layers to handle the time series prediction task.
Evaluation: The model is evaluated using RMSE (Root Mean Squared Error) and R-squared metrics.
Visualization: The model's predictions are compared with actual values in graphical format.

## Requirements
Before running the code, ensure that the following Python libraries are installed:

torch
pandas
numpy
matplotlib
scikit-learn
You can install the required libraries using pip:
pip install torch pandas numpy matplotlib scikit-learn

## Code Walkthrough
### 1. Data Preprocessing
The dataset is first split into features (x) and the target (y), followed by a 70% training and 30% test split. Each dataset is normalized using MinMaxScaler.
### 2. Transforming Data for PyTorch
After preprocessing, the data is converted to PyTorch tensors.
### 3. Model Definition
A custom Transformer model is defined with multi-head self-attention layers, followed by a fully connected layer to generate the predictions.
### 4. Training the Model
The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function. During training, the loss is calculated, and the model weights are updated using backpropagation.
### 5. Model Evaluation
After training, the model is evaluated on the test dataset using RMSE and R-squared metrics.
### 6. R-squared Calculation
R-squared is calculated to assess the goodness of fit between the predicted and actual values.
### 7. Prediction and Visualization
The predictions are made using the trained model and then plotted alongside the actual values for visual comparison.
### 8. Combined Data Plot
The results from both the training and testing datasets are combined, and a graph is plotted to show the model's predictions.

## Results and Conclusion
The model demonstrates good performance in predicting time series data, with reasonable error metrics such as RMSE and R-squared values.
The plotted results show how the model's predictions compare with the actual data, indicating the effectiveness of the Transformer model for time series forecasting.
