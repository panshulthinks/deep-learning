import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### finding the min, max, mean, sum, etc
x = torch.arange(0, 100, 10)
print(x)
print("min:", x.min())
print("max:", x.max())
print("mean:", x.type(torch.float32).mean())
print("sum:", x.sum())
print("the value where the max occurs:", x.argmax())
print("product of all the elements:", x.prod())

### reshaping, view, stacking, squeezing and unsqueezing tensors
y = torch.arange(0, 10)
print("shape of y:", y.shape)
y_reshaped = y.reshape(1, 10)
print("reshaped y:", y_reshaped)
print("shape of y reshaped", y_reshaped.shape)

# view
z = y.view(1, 10)
print("view of z:", z)

# stcaking
y_stacked = torch.stack([y, y, y, y], dim=1)
print("stacked y:", y_stacked)

#squeezing
tensor1 = torch.tensor([[1, 3, 1]])
print("squeezed tensor1:", tensor1.squeeze())

# permute
tensor2 = torch.tensor([[[3, 6, 9]]])
print("permuted tensor2:", tensor2.permute(1, 0, 2))