import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn


weight = 0.7 #this is a known parameter
bias = 0.3  #this is a known parameter

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias

train_split = int(len(X) * 0.8)
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define model parameters
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))
        # forward method which defines the computation
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weight * x +self.bias# this is linear regression formula

def plot_prediction(train_data = X_train,
                    train_labels = Y_train,
                    test_data = X_test,
                    test_labels = Y_test,
                    predictions = None):
     plt.figure(figsize=(10, 6))
#plotting training data
     plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
#plotting testing data
     plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
#plotting predictions
     if predictions is not None:
     #plot if they exist
         plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
     plt.legend(prop={"size": 14})
     plt.show()     
torch.manual_seed(42) # for reproducibility
# create an instance of the model
model_0 = LinearRegressionModel()
# print model parameters
print(list(model_0.parameters())) # returns a list of model parameters

with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)
print(Y_test)

plot_prediction(predictions = y_preds)