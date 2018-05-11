#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 19:15:45 2018

@author: andrewburger
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


pca_scores = []
nonpca_scores = []


# Logistic Regression with PCA and NON PCA With 10 data sets





def getScores1DS(data):   # this function is if your data is not split
    
    
    data = np.genfromtxt(data, delimiter = ',')


    X = data[:,:-1].copy()
    y = data[:,-1].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

    pca_train = PCA(n_components = 3)
    pca_test = PCA(n_components = 3)


    Xpca_train = pca_train.fit_transform(X_train)
    Xpca_test = pca_test.fit_transform(X_test)


    lr_pca = LogisticRegression()
    lr_nonpca = LogisticRegression()


    lr_pca.fit(Xpca_train,y_train)
    lr_nonpca.fit(X_train,y_train)

    pca_score = lr_pca.score(Xpca_test,y_test)
    nonpca_score = lr_nonpca.score(X_test,y_test)
    
        

    return pca_score,nonpca_score


def getScores2DS(dataTrain,dataTest): # if it is split
    
    data_test = np.genfromtxt(dataTest, delimiter = ',')
    data_train = np.genfromtxt(dataTrain, delimiter = ',')

    X_train = data_train[:,:-1].copy()
    y_train = data_train[:,-1].copy()

    X_test = data_test[:,:-1].copy()
    y_test = data_test[:,-1].copy()
    
    pca_train = PCA(n_components = 3)
    pca_test = PCA(n_components = 3)


    Xpca_train = pca_train.fit_transform(X_train)
    Xpca_test = pca_test.fit_transform(X_test)


    lr_pca = LogisticRegression()
    lr_nonpca = LogisticRegression()


    lr_pca.fit(Xpca_train,y_train)
    lr_nonpca.fit(X_train,y_train)

    pca_score = lr_pca.score(Xpca_test,y_test)
    nonpca_score = lr_nonpca.score(X_test,y_test)
    
        

    return pca_score,nonpca_score


#bank data
pcascore0,score0 = getScores1DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/bank.csv")

#  Breast Cancer data set

pcascore1,score1 = getScores1DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/breast-cancer.csv")

# congressional voting data set

pcascore2, score2 = getScores1DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/congressional-voting.csv")

#credit approval data set

pcascore3, score3 = getScores1DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/credit-approval.csv")

#monks1 Data Set

pcascore4, score4 = getScores2DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/monks-1_train.csv",
                                         "/Users/andrewburger/Desktop/School/ECE523/DataSets/monks-1_test.csv")

#monks 2 Data Set

pcascore5, score5 = getScores2DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/monks-2_train.csv",
                                 "/Users/andrewburger/Desktop/School/ECE523/DataSets/monks-2_test.csv")

#monks 3 data set

pcascore6, score6 = getScores2DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/monks-3_train.csv",

                                 "/Users/andrewburger/Desktop/School/ECE523/DataSets/monks-3_test.csv")
#parkinsons data set

pcascore7, score7 = getScores1DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/parkinsons.csv")

#pima data set

pcascore8, score8 = getScores1DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/pima.csv")

#titanic data set

pcascore9, score9 = getScores1DS("/Users/andrewburger/Desktop/School/ECE523/DataSets/titanic.csv")


pca_scores.append(pcascore0)
pca_scores.append(pcascore1)
pca_scores.append(pcascore2)
pca_scores.append(pcascore3)
pca_scores.append(pcascore4)
pca_scores.append(pcascore5)
pca_scores.append(pcascore6)
pca_scores.append(pcascore7)
pca_scores.append(pcascore8)
pca_scores.append(pcascore9)

nonpca_scores.append(score0)
nonpca_scores.append(score1)
nonpca_scores.append(score2)
nonpca_scores.append(score3)
nonpca_scores.append(score4)
nonpca_scores.append(score5)
nonpca_scores.append(score6)
nonpca_scores.append(score7)
nonpca_scores.append(score8)
nonpca_scores.append(score9)

results = np.column_stack((pca_scores,nonpca_scores))    

print(results)



















