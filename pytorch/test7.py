import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

weight = 0.7 #this is a known parameter
bias = 0.3  #this is a known parameter

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias

print(X[:10])
print(Y[:10])
print(len(X))
print(len(Y))

#splitting data and makingx
train_split = int(len(X) * 0.8)
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

print(len(X_train))
print(len(Y_train))
print(len(X_test))
print(len(Y_test))

#visualizing the data
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
plot_prediction()