import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

# building a training loop
# A couple of things we need in a training loop: 
# 0. Loop through the data 
# 1. Forward pass (this involves data moving through our model's forward()` functions) to make predictions on data also called forward propagation 
# 2. Calculate the loss (compare forward pass predictions to ground truth labels) 
# 3. Optimizer zero grad 
# 4. Loss backward "-move" backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss (**backpropagation* 
# 5. Optimizer step - use the optimizer to adjust our model's parameters to try and improve the loss (**gradient descent**)

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


l1_loss = nn.L1Loss() #L1 is known as mean absolute error

# optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) # Stochastic Gradient Descent 




#track different values
epoch_count = []
train_loss_values = []
test_loss_values = []




















epochs = 100 #how many times to loop through the data
for epoch in range(epochs):
    # 1. Forward pass
    y_preds = model_0(X_train)
    
    # 2. Calculate the loss
    loss = l1_loss(y_preds, Y_train)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()
    
    # 4. Loss backward
    loss.backward()
    
    # 5. Optimizer step
    optimizer.step()
    
    model_0.eval() # set the model to evaluation mode
    with torch.inference_mode(): # turns of gradient tracking
         # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = l1_loss(test_pred, Y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
    
      # Print out what's happening
    if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

            print(model_0.state_dict()) # print model parameters

with torch.inference_mode():
    y_preds_new = model_0(X_test)

# Plot the training and test loss values
#plot_prediction(predictions=y_preds)
plot_prediction(predictions=y_preds_new)

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(); 
plt.show()       