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
```bash
pip install torch pandas numpy matplotlib scikit-learn
```

## Code Walkthrough
### 1. Data Preprocessing
The dataset is first split into features (x) and the target (y), followed by a 70% training and 30% test split. Each dataset is normalized using MinMaxScaler.
```python
x = df.iloc[:,0:-1]
y = df.iloc[:,-1:]

# Training/Test ratio = 7:3
Train_ratio = 0.7
Test_ratio = 0.3

# Data Division
train_x = x.iloc[0:int(len(df)*Train_ratio),:]
train_y = y.iloc[0:int(len(df)*Train_ratio),:]
test_x = x.iloc[int(len(df)*Train_ratio):,:]
test_y = y.iloc[int(len(df)*Train_ratio):,:]

# Normalizing
minmax = MinMaxScaler()
train_x = minmax.fit_transform(train_x)
train_y = minmax.fit_transform(train_y)
test_x = minmax.fit_transform(test_x)
test_y = minmax.fit_transform(test_y)
```
### 2. Transforming Data for PyTorch
After preprocessing, the data is converted to PyTorch tensors.
```python
train_x_tensor = Variable(torch.Tensor(train_x))
train_y_tensor = Variable(torch.Tensor(train_y))
test_x_tensor = Variable(torch.Tensor(test_x))
test_y_tensor = Variable(torch.Tensor(test_y))

# Reshaping for the Transformer input format
train_x_tensor_final = torch.reshape(train_x_tensor, (train_x_tensor.shape[0], 1, train_x_tensor.shape[1]))
train_y_tensor_final = torch.reshape(train_y_tensor, (train_y_tensor.shape[0], 1, train_y_tensor.shape[1]))
test_x_tensor_final = torch.reshape(test_x_tensor, (test_x_tensor.shape[0], 1, test_x_tensor.shape[1]))
test_y_tensor_final = torch.reshape(test_y_tensor,(test_y_tensor.shape[0], 1, test_y_tensor.shape[1]))
```
### 3. Model Definition
A custom Transformer model is defined with multi-head self-attention layers, followed by a fully connected layer to generate the predictions.
```python
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead, dropout):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(input_size, 1)

    def forward(self, src, tgt):
        x = self.transformer(src, tgt)
        x = self.fc(x[:, -1, :])
        return x
```
### 4. Training the Model
The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function. During training, the loss is calculated, and the model weights are updated using backpropagation.
```python
# Hyperparameters
embed_dim = 20
input_size = train_x_tensor_final.shape[2]
hidden_size = 20
num_layers = 2
nhead = 2
dropout = 0.2

# Initialize Model
model = TransformerModel(embed_dim, hidden_size, num_layers, nhead, dropout)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 500
for epoch in range(num_epochs):
    outputs = model(train_x_tensor_final, train_x_tensor_final)
    loss = criterion(outputs, torch.reshape(train_y_tensor_final, (train_y_tensor_final.shape[0], train_y_tensor_final.shape[1])))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```
### 5. Model Evaluation
After training, the model is evaluated on the test dataset using RMSE and R-squared metrics.
```python
# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(test_x_tensor_final, test_x_tensor_final)
    test_loss = criterion(test_outputs, torch.reshape(test_y_tensor_final, (test_y_tensor_final.shape[0], test_y_tensor_final.shape[1])))
    print(f"Test Loss: {test_loss.item():.4f}")

    # RMSE calculation
    test_rmse = torch.sqrt(test_loss).item()
    print(f"Test RMSE: {test_rmse:.4f}")
```
### 6. R-squared Calculation
R-squared is calculated to assess the goodness of fit between the predicted and actual values.
```python
def r_squared(predicted, actual):
    predicted_mean = np.mean(predicted)
    total_variation = np.sum((actual - predicted_mean) ** 2)
    residual_variation = np.sum((actual - predicted) ** 2)
    return 1 - (residual_variation / total_variation)

r2_train = r_squared(train_outputs, train_y_tensor_final)
r2_test = r_squared(test_outputs, test_y_tensor_final)

print(f"Train R-squared: {r2_train:.4f}")
print(f"Test R-squared: {r2_test:.4f}")
```
### 7. Prediction and Visualization
The predictions are made using the trained model and then plotted alongside the actual values for visual comparison.
```python
predicted = minmax.inverse_transform(np.concatenate((test_x, test_outputs.numpy()), axis=1))[:, -1]
actual = minmax.inverse_transform(np.concatenate((test_x, test_y.reshape(-1, 1)), axis=1))[:, -1]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(actual, label='Actual Data')
plt.plot(predicted, label='Predicted Data')
plt.title('Time Series Prediction')
plt.legend()
plt.show()
```
### 8. Combined Data Plot
The results from both the training and testing datasets are combined, and a graph is plotted to show the model's predictions.
```python
import copy
data_combined_x = np.concatenate((train_x, test_x), axis=0)
data_combined_y = np.concatenate((train_y, test_y), axis=0)
PlotEstimated = np.empty_like(np.concatenate((data_combined_x, data_combined_y), axis=1))
PlotEstimated[:, :] = np.nan
testPlotEstimated = copy.deepcopy(PlotEstimated)
result = predicted.reshape(34, 1)
testPlotEstimated[79:, -1:] = result

plt.figure(figsize=(10, 5))
plt.plot(minmax.inverse_transform(data_combined_y))
plt.plot(testPlotEstimated)
plt.show()
```
## Results and Conclusion
The model demonstrates good performance in predicting time series data, with reasonable error metrics such as RMSE and R-squared values.
The plotted results show how the model's predictions compare with the actual data, indicating the effectiveness of the Transformer model for time series forecasting.
