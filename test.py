"""
Description: A script ment to test my implementation of the Adaline neuron.
Date: October 2022
Author: Aleksa Zatezalo
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import adaline as adaline
from matplotlib.colors import ListedColormap

# Getting Iris Photos
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('From URL: ', s)
df = pd.read_csv(s, header=None, encoding='utf-8')

# Getting first 100 Iris
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Extract sepal length and petal length
X = df.iloc[0:100, [0,2]].values

# Testing Learning Rates