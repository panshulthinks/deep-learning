import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles

n_samples = 1000
x, y = make_circles(n_samples, noise=0.1, factor=0.5) 
print(x.shape)
print(y.shape)

