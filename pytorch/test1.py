import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

scalar = torch.tensor(7)
scalar
scalar.ndim
scalar.item()

print("scalar:", scalar)
print("scalar dim:", scalar.ndim)
print("scalar item:", scalar.item())

vector = torch.tensor([7, 7])
print("vector:", vector)

vector.ndim
vector.shape
print("vector dim:", vector.ndim)
print("vector shape:", vector.shape)

MATRIX = torch.tensor([[6, 7],
                       [67, 69]])
print("matrix:", MATRIX)
print("matrix dim:", MATRIX.ndim)
print("first row:", MATRIX[0])
print("second row:", MATRIX[1])

TENSOR = torch.tensor([[[2, 4, 8],
                        [4, 6, 3],
                        [8, 5, 7]]])
print(TENSOR)
print("Tensor dim:", TENSOR.ndim)
print("Tensor shape:", TENSOR.shape)
print(TENSOR[0])

TENSOR1 = torch.tensor([[[[[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]]]])
print(TENSOR1)
print("Tensor dim:", TENSOR1.ndim)
print("Tensor shape:", TENSOR1.shape)
print(TENSOR1[0])


TENSOR2 = torch.tensor([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]]])

print(TENSOR2)
print("Tensor dim:", TENSOR2.ndim)    
print("Tensor shape:", TENSOR2.shape) 
