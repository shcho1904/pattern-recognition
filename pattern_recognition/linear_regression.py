# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:06:27 2022

@author: shcho
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

data_path = 'C:/Users/shcho/Desktop/course material/2022-2R/pattern_recognition/Boston_house.csv'

var1 = pd.read_csv(data_path, 
                 usecols=['CRIM', 'RAD'])
var2 = pd.read_csv(data_path, 
                 usecols=['CRIM', 'RAD', 'RM', 'AGE'])
var3 = pd.read_csv(data_path, 
                 usecols=['CRIM', 'RAD', 'RM', 'AGE', 'DIS', 'B', 'INDUS', 'LSTAT', 'NOX', 'ZN'])
price = pd.read_csv(data_path,
                    usecols=['Target'])

var1_train = torch.FloatTensor(var1.iloc[0:int(var1.shape[0]*0.8)].to_numpy())
var1_test = torch.FloatTensor(var1.iloc[int(var1.shape[0]*0.8):var1.shape[0]].to_numpy())

var2_train = torch.FloatTensor(var2.iloc[0:int(var2.shape[0]*0.8)].to_numpy())
var2_test = torch.FloatTensor(var2.iloc[int(var2.shape[0]*0.8):var2.shape[0]].to_numpy())

var3_train = torch.FloatTensor(var3.iloc[0:int(var3.shape[0]*0.8)].to_numpy())
var3_test = torch.FloatTensor(var3.iloc[int(var3.shape[0]*0.8):var3.shape[0]].to_numpy())

price_train = torch.FloatTensor(price.iloc[0:int(price.shape[0]*0.8)].to_numpy())
price_test = torch.FloatTensor(price.iloc[int(price.shape[0]*0.8):price.shape[0]].to_numpy())

#var1 regression
w_var1 = torch.zeros((2,1), requires_grad=True)
b_var1 = torch.zeros(1, requires_grad=True)

w_var2= torch.zeros((4,1), requires_grad=True)
b_var2 = torch.zeros(1, requires_grad=True)

w_var3 = torch.zeros((10,1), requires_grad=True)
b_var3 = torch.zeros(1, requires_grad=True)

optimizer1 = optim.SGD([w_var1, b_var1], lr=1e-5)
optimizer2 = optim.SGD([w_var2, b_var2], lr=1e-5)
optimizer3 = optim.SGD([w_var3, b_var3], lr=1e-6)

for epoch in range(1000):
    
    # H(x)
    hypothesis1 = var1_train.matmul(w_var1) + b_var1
    hypothesis2 = var2_train.matmul(w_var2) + b_var2
    hypothesis3 = var3_train.matmul(w_var3) + b_var3
    
    # cost
    cost1 = torch.mean((hypothesis1 - price_train)**2)
    cost2 = torch.mean((hypothesis2 - price_train)**2)
    cost3 = torch.mean((hypothesis3 - price_train)**2)
    
    #optimize
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    
    cost1.backward()
    cost2.backward()
    cost3.backward()
    
    optimizer1.step()
    optimizer2.step()
    optimizer3.step()
    
predict_price1 = (var1_test.matmul(w_var1) + b_var1).detach().numpy()
predict_price2 = (var2_test.matmul(w_var2) + b_var2).detach().numpy()
predict_price3 = (var3_test.matmul(w_var3) + b_var3).detach().numpy()

fig, axs = plt.subplots(3)
axs[0].scatter(price_test, predict_price1)
axs[0].set_title('variable 2')

axs[1].scatter(price_test, predict_price2)
axs[1].set_title('variable 4')

axs[2].scatter(price_test, predict_price3)
axs[2].set_title('variable 10')