# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:09:09 2018

@author: AARUSHI
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc 



filename = './data/Q1/MNIST_Subset.h5'
f = h5py.File(filename, 'r')

print(len(list(f.keys())))
# List all groups
coord = list(f.keys())[0]
label = list(f.keys())[1]

# Get the data
data1 = list(f[coord])
data2 = list(f[label])

X_data = []
for i in range(0,14251):
    X_data.append(np.ravel(data1[i]))
    
X_data = np.asarray(X_data)
Y_data = np.asarray(data2)

Y_data[Y_data==7] = 0
Y_data[Y_data==9] = 1
#store = np.reshape(X_data[0],(X_data[0].shape[0],1))
num_nodes = [784,100,50,50,2]
model = Neural_Network(5,num_nodes,1)
model.train(1, X_data, Y_data, 1,0.001, 'model.pkl')
#y_predicted,a = model.predict('model.pkl',X_data)
model.check_accuracy('model.pkl',X_data,Y_data)