#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 22:14:56 2018

@author: andrewburger
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from cvxopt import matrix
from cvxopt import solvers
from sklearn.svm import SVC

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Train source weights
def train_source(alpha, X, y, n):
    
    w = np.random.randn(2)
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

#solve quadratic programming problem for alpha
    

def SVM_Solve(B,C,X,Y,W):
    
    m = len(X)
    tempP = np.dot(Y,np.transpose(Y)) * np.dot(X,np.transpose(X)) 
    P = matrix(tempP)
    tempQ = B*np.dot(W,np.transpose(X)) - 1.0
    Q = matrix(tempQ)
    tempG = np.asarray([-1.0,1.0])
    G = matrix(np.transpose(tempG))
    tempH = np.asarray([0.0,C])
    H = matrix(np.transpose(tempH))
   # A = matrix(np.transpose(Y))
    A = np.reshape((Y.T), (1,m))
    A = matrix(A)
    b = matrix([0.0])
    
    sol = solvers.qp(P,Q,G,H,A,b)
    
    return sol


#correct svm
def svm(B,c,X,Y,W):
      
      m = len(X)
      P = matrix(np.dot(Y, Y.T) * np.dot(X, X.T))
      tempQ = B*np.dot(W,np.transpose(X)) - 1
      q = matrix(tempQ)
      g1 = np.asarray(np.diag(np.ones(m) * -1))
      g2 = np.asarray(np.diag(np.ones(m)))
      G = matrix(np.append(g1, g2, axis=0))
     # h = matrix(np.append(np.zeros(1), (np.ones(1)*c), axis = 0))
      h = matrix(np.append(np.zeros(m), (np.ones(m) * c), axis =0))
      A = np.reshape((Y.T), (1,m))
      b = matrix([0.0])
      

      A = matrix(A)
      

      sol = solvers.qp(P, q, G, h, A, b)
      return sol
  
    
def predict(w, w0,X,n):
    
    wt = np.transpose(w)
    yhat = []
    
    for c in range(n):
    
        Xn = X[c,:]
        yhat.append(np.dot(wt,Xn)+w0)
        
        
    
    return yhat
    

#getting source data

data_test = np.genfromtxt("/Users/andrewburger/Desktop/School/ECE523/DataSets/source_test.csv", delimiter = ',')
data_train = np.genfromtxt("/Users/andrewburger/Desktop/School/ECE523/DataSets/source_train.csv", delimiter = ',')

X_train_source = data_train[:,:-1].copy()
y_train_source = data_train[:,-1].copy()

X_test_source = data_test[:,:-1].copy()
y_test_source = data_test[:,-1].copy()

y_test_source = np.array(y_test_source)

data_test_target = np.genfromtxt("/Users/andrewburger/Desktop/School/ECE523/DataSets/target_test.csv", delimiter = ',')
data_train_target = np.genfromtxt("/Users/andrewburger/Desktop/School/ECE523/DataSets/target_train.csv", delimiter = ',')

X_train_target= data_train_target[:,:-1].copy()
y_train_target = data_train_target[:,-1].copy()

X_test_target = data_test_target[:,:-1].copy()
y_test_target = data_test_target[:,-1].copy()

y_test_target = np.array(y_test_target)



#training source weights using gradient descent


Wsource,Wsource0 = train_source(.03,X_train_source, y_train_source, 200)

solver = svm(1.0,1.0,X_train_target, y_train_target,Wsource)
alpha = solver['x']
print (alpha)


Wtarget = alpha*y_train_target*X_train_target # not sure getting dimensionality issues when solving for Wtrain

y_prediction = predict(Wtarget,Wtarget0,X_test_source,400)
y_prediction=np.array(y_prediction)

print(np.sum((y_prediction>0)==(y_test_target==1))/1.0/len(y_test_target))



