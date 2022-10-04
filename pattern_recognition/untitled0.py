# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:44:45 2022

@author: shcho
"""

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)

"""PCA Algorithm"""
