#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:47:23 2018

@author: andrewburger
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp



def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Train using SGD pass in training Data set
def train(alpha, X, y, n):
    
    w = np.random.randn(9)
    w0 = 1.
   
    T= 4000
    
    j = np.random.permutation(n)
    X = X[j,:]
    y = y[j]
    
    # add in w0
    
    for k in range(T):
        #np.random.shuffle(X)
       length = len(X)
       indices = np.arange(length)
       np.random.shuffle(indices)
       X = X[indices]
       y = y[indices]
       for c in range(n):
            
            Xn = X[c,:]
            yn = y[c]
            
            delta = alpha * (sigmoid(np.dot(w,Xn)+w0)- yn)
            
            w = w - delta*Xn
            w0 = w0 - delta
            
    
    return w, w0

# Test pass in Testing Data set
def predict(w, w0,X,n):
    
    wt = np.transpose(w)
    yhat = []
    
    for c in range(n):
    
        Xn = X[c,:]
        yhat.append(sigmoid(np.dot(wt,Xn)+w0))
        
        
    
    return yhat


data_test = np.genfromtxt("C:\\Users\\Elena Burger\\Desktop\\ECE523\\Data\\CSV\\BreastCancerTest.csv", delimiter = ',')
data_train = np.genfromtxt("C:\\Users\\Elena Burger\\Desktop\\ECE523\\Data\\CSV\\BreastCancerTraining.csv", delimiter = ',')

X_train = data_train[:,:-1].copy()
y_train = data_train[:,-1].copy()

X_test = data_test[:,:-1].copy()
y_test = data_test[:,-1].copy()

y_test = np.array(y_test)


#train 
w, w0 = train(.01,X_train,y_train,228)

#test
y_prediction = predict(w,w0,X_test,58)
y_prediction=np.array(y_prediction)

print(np.sum((y_prediction>0.5)==(y_test==1))/1.0/len(y_test))
