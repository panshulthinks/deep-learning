import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.__version__)


print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)