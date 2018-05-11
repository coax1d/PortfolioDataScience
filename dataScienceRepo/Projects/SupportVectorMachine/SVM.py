#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:22:08 2018

@author: andrewburger
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt




plt.close()

X1 = np.random.multivariate_normal([-1,-1], [[1, -.25], [-.25, 1]], 1000).T
X2 = np.random.multivariate_normal([1, 1], [[1, -.25], [-.25, 1]], 1000).T
X = X1 + X2
X = np.asarray([list(a) for a in zip(X1[0],X1[1])] + [list(a) for a in zip(X2[0],X2[1])])
y = np.asarray([-1]*len(X1[0])+[1]*len(X2[0]))


#rbf
firstClassifier = svm.SVC(C = 1, kernel = 'rbf', gamma = .5)

firstClassifier.fit(X,y)
print(firstClassifier.fit(X,y).score(X,y))

fig, axis = plt.subplots(1, figsize=(10, 5))
xx, yy = np.meshgrid(np.linspace(-4, 4, 400), np.linspace(-5, 5, 400))

Z = firstClassifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axis.contourf(xx,yy, Z, alpha=0.3)
axis.scatter(X[:, 0], X[:, 1], c=y, s=12, alpha=0.4, edgecolors='black')

#linear
secondClassifier = svm.SVC(C = 1, kernel = 'linear', gamma = .5)

secondClassifier.fit(X,y)


figure2, axis2 = plt.subplots(1, figsize=(10, 5))
xx2, yy2 = np.meshgrid(np.linspace(-4, 4, 400), np.linspace(-5, 5, 400))

Z_ = secondClassifier.decision_function(np.c_[xx2.ravel(), yy2.ravel()])
Z_ = Z_.reshape(xx2.shape)

axis2.contourf(xx2,yy2, Z_, alpha=0.3)
axis2.scatter(X[:, 0], X[:, 1], c='b', s=12, alpha=.4, edgecolors='black')
print(secondClassifier.fit(X,y).score(X,y))



