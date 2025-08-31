import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### random tensors


random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)


zeroes = torch.zeros(3, 4)
print(zeroes)

multiply = zeroes*random_tensor
print("Matrix mulitplication:", multiply)

ones = torch.ones(3, 4)
print(ones)
ones.dtype
print(ones.dtype)

print(torch.arange(1, 11))
print(torch.arange(0, 100, 5))#here 5 means that we step it up by 5


### tensor datatypes


float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype = None,
                               device = None,
                               requires_grad= False)
print(float_32_tensor)

# lets convert our float32tensor to float16
float_16_tensor = float_32_tensor.type(torch.half)
print(float_16_tensor)

#lets make a int 32 tensor
int_32_tensor = torch.tensor([[1, 3 , 6]], dtype= torch.int32)
print(int_32_tensor)


floatmult = float_32_tensor * float_16_tensor
print("floatmultiplication:", floatmult)


#lets get information from our tensors(tensor attributes)
#to get datatype from a tensor we can use tensor.dtype
#to get shape from a tensor we can use tensor.shape
#to get device from a tensor we can use tensor.device

some_tensor = torch.rand(3, 4)
some_tensor = some_tensor.type(torch.half)
print(some_tensor)
print(f"dataype of the some tensor: {some_tensor.dtype}")
print(f"shape of the some tensor: {some_tensor.shape}")
print(f"device of our tensors: {some_tensor.device}")

### manipulating tensors(tensors opertions)
tensor = torch.tensor([1, 7, 4])
tensor += 100
print(f"tensor addition: {tensor}")


tensor *= 10
print("tensor multiplication", tensor)

tensor -= 69
print("tensor subtraction:", tensor)

tensor1 = torch.tensor([3, 6, 9])
elementwise_multiplication = tensor1 * tensor1
print("elementwise multi:", elementwise_multiplication)

### lets do a matrix multiplication now
### we will be using pytorchs matmul function
### we can also transpose a matrix using .t attribute