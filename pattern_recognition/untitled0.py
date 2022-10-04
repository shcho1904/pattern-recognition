# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:44:45 2022

@author: shcho
"""

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import pandas as pd
from sklearn.preprocessing import StandardScaler

#load data from mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
INF = 987654321.0
test_case = 200
train_num = 1000

#pull 20 samples for training
X_train = X_train[0:train_num]

#pick 5 samples for testing
X_test = X_test[0:test_case]

#show one of sample images
'''
image = X_train[4]
fig = plt.figure()
plt.imshow(image)
plt.show()
'''

#flatten data
A = X_train.reshape(-1, 28*28)
test = X_test.reshape(-1, 28*28)

#get mean of training data
A = StandardScaler().fit_transform(A)
test = StandardScaler().fit_transform(test)


#compute covariance matrix
Cov_mat = A.T.dot(A)

#compute eigenvalues and eigenvectors
values, vectors = eigh(Cov_mat)
vectors = vectors.T
new_vec = vectors.dot(A.T)

#calculate weight per each eigenvector
ohm_test = test.dot(new_vec)
ohm_train = A.dot(new_vec)

#calculate distance
best_arr = []

for i in range(X_test.shape[0]):
    best_error = INF
    best_idx = 0
    error = 0.0
    for j in range(X_train.shape[0]):
        error = np.sum((ohm_test[i] - ohm_train[j])**2)
        if(best_error > error):
            best_error = error
            best_idx = y_train[j]
    best_arr.append(best_idx)

num_correct = np.sum(y_test[0:test_case] == best_arr)
accuracy = float(num_correct) / test_case
print(accuracy*100)
