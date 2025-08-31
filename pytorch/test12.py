import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

# create a data
weight = 0.7  # this is a known parameter
bias = 0.3    # this is a known parameter

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias

# split data into train and test sets
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

# define a linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear_layer(x)
    
torch.manual_seed(42)
model_1 = LinearRegressionModel()
model_1, model_1.state_dict()





# train the model
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(lr=0.01, params=model_1.parameters())

epochs = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
model_1.to(device)
X_train = X_train.to(device)
X_test = X_test.to(device)
Y_train = Y_train.to(device)
Y_test = Y_test.to(device)

for epoch in range(epochs):
     model_1.train()
     #forward pass
     y_pred = model_1(X_train)
     #calculate loss
     loss = loss_fn(y_pred, Y_train)   
     #zero gradients
     optimizer.zero_grad()
     #backward pass 
     loss.backward()
     #optimizer step
     optimizer.step()
     #testing
     model_1.eval()
     with torch.inference_mode():
            y_test_pred = model_1(X_test)
            test_loss = loss_fn(y_test_pred, Y_test)

if epoch % 10 == 0:
      print(f"Epoch: {epoch} | MAE Train Loss: {loss:.5f} | MAE Test Loss: {test_loss:.5f}")
from pprint import pprint
print("The model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")



# Evaluate the model
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test)

def plot_prediction(train_data=X_train,
                   train_labels=Y_train,
                   test_data=X_test,
                   test_labels=Y_test,
                   predictions=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(train_data.cpu(), train_labels.cpu(), c="b", s=4, label="Training data")
    plt.scatter(test_data.cpu(), test_labels.cpu(), c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data.cpu(), predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()

        
plot_prediction(predictions=y_preds.cpu())











