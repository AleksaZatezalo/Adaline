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
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,4))

# Learning Rate 0.1
ada1 = adaline.AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) +1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - learning rate 0.1')

# Learning Rate 0.0001
ada2 = adaline.AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) +1), np.log10(ada2.losses_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Mean squared error)')
ax[1].set_title('Adaline - learning rate 0.0001')

plt.show()

# Standardizing Data
X_std = np.copy(X)
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()


# Graphing Standardized Data
ada_gd = adaline.AdalineGD(n_iter=20, eta=0.5)
ada_gd.fit(X_std, y)
plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker="o")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.tight_layout()
plt.show()