#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 06:43:57 2018

@author: andrewburger
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


filePath = "/Users/andrewburger/Desktop/School/ECE523/DataSets/DataSets1/"
data = ["bank.csv",
        "breast-cancer.csv",
        "congressional-voting.csv",
        "credit-approval.csv",
        "cylinder-bands.csv",
        "hepatitis.csv",
        "ionosphere.csv",
        "mammographic.csv",
        "mushroom.csv",
        "parkinsons.csv",
        "pima.csv",
        "pittsburg-bridges-T-OR-D.csv",
        "planning.csv",
        "ringnorm.csv",
        "titanic.csv"]

baggingAccuracies = []
adaBoostAccuracies = []

def getData(dataSet):
        
    data1 = np.genfromtxt(filePath+dataSet, delimiter = ',')


    X = data1[:,:-1].copy()
    y = data1[:,-1].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)
        
    return X_train, X_test, y_train, y_test
    
    

def baggingTestAccuracy():
    
    
    for i, dataSample in enumerate(data):
        
        X_train,X_test, y_train, y_test = getData(dataSample)
        
        for k in range(2,150,25):
            
            clf = BaggingClassifier(n_estimators = k, random_state = 87)
            clf.fit(X_train, y_train)
            
            score = clf.score(X_test,y_test)
            baggingAccuracies.append(score)

def adaBoostTestAccuracy():

    for i, dataSample in enumerate(data):
        
        X_train,X_test, y_train, y_test = getData(dataSample)
        
        for k in range(2,150,25):
            
            clf = AdaBoostClassifier(n_estimators = k, random_state = 87)
            clf.fit(X_train, y_train)
            
            score = clf.score(X_test,y_test)
            adaBoostAccuracies.append(score)
        
 
baggingTestAccuracy()
adaBoostTestAccuracy()


#convert to np array for plotting
baggingAccuracies = np.asarray(baggingAccuracies)
adaBoostAccuracies = np.asarray(adaBoostAccuracies)

yPointsForBagging = np.reshape((baggingAccuracies), (1,len(baggingAccuracies)))
yPointsForAda = np.reshape((adaBoostAccuracies), (1,len(adaBoostAccuracies)))

#num of classifiers
xPoints =          [2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127,
                    2, 27, 52, 77, 102, 127]

xPoints = np.asarray(xPoints)
xPoints = np.reshape((xPoints), (1,len(xPoints)))

baggingPoints = np.vstack((xPoints,yPointsForBagging))
adaBoostPoints = np.vstack((xPoints,yPointsForBagging))

print(baggingPoints.T)
print(adaBoostPoints.T)
    







