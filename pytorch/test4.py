import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### indexing

x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)
print("accessing the first element of the matrix", x[0, 0, 0])
print("accessing the second element of the matrix", x[0, 0, 1])
print("lets see the 2 row 2 element of the matrix", x[0, 1, 1])


