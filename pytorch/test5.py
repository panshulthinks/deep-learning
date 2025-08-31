import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

array = np.arange(.0, 8.0, 2.0)
tensor = torch.from_numpy(array)
print(array)
print(tensor)
print(array.dtype)
print(tensor.dtype)

#tensor to array
tensor1 = torch.ones(7)
array1 = tensor1.numpy()
print(tensor1)
print(array1)
print(tensor1.dtype)
print(array1.dtype)


### reproductibility
random_tensor1 = torch.rand(3, 4)
random_tensor2 = torch.rand(3, 4)

print(random_tensor1)
print(random_tensor2)
print(random_tensor1 == random_tensor2)


### random seed
RANDOM_SEED = 7
torch.manual_seed(RANDOM_SEED)
randomtensor3 = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED)
randomtensor4 = torch.rand(3,4)
print(randomtensor3)
print(randomtensor4)
print(randomtensor3 == randomtensor4)

