import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define model parameters
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))
        # forward method which defines the computation
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the input data
        return self.weight * x +self.bias# this is linear regression formula
    
torch.manual_seed(42) # for reproducibility
# create an instance of the model
model_0 = LinearRegressionModel()
# print model parameters
print(list(model_0.parameters())) # returns a list of model parameters



# loss function is a parameter to measure how wrong our predictions are
l1_loss = nn.L1Loss() #L1 is known as mean absolute error

# optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) # Stochastic Gradient Descent 

