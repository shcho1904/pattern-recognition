# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:44:45 2022

@author: shcho
"""

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

#load data from mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
INF = 987654321.0
test_case = 1000
train_num = 15000
eig_num = 40

#pull 20 samples for training
X_train = X_train[0:train_num]

#pick 5 samples for testing
X_test = X_test[0:test_case]

#show one of sample images

image = X_train[4]
fig = plt.figure()
plt.imshow(image)
plt.show()

#flatten data
A = X_train.reshape(-1, 28*28)
test = X_test.reshape(-1, 28*28)

#get mean of training data
A = StandardScaler().fit_transform(A)
test = StandardScaler().fit_transform(test)
A = A.T
test = test.T

#compute covariance matrix
Cov_mat = np.matmul(A.T, A)

#compute eigenvalues and eigenvectors
values, vectors = eigh(Cov_mat, eigvals=(784-eig_num,783))
new_vec = A.dot(vectors)

#calculate weight
train_weight = new_vec.T.dot(A).T


#test
test_weight = new_vec.T.dot(test).T

best_arr = []

for i in range(X_test.shape[0]):
    best_error = INF
    best_idx = 0
    error = 0.0
    for j in range(X_train.shape[0]):
        #calculate distance between test and train data
        error = np.sum((test_weight[i] - train_weight[j])**2)
        if(best_error > error):
            #find best index has least error
            best_error = error
            best_idx = y_train[j]
    best_arr.append(best_idx)

#calculate accuracy
num_correct = np.sum(y_test[0:test_case] == best_arr)
accuracy = float(num_correct) / test_case

#print accuracy(%)
print(accuracy*100)



