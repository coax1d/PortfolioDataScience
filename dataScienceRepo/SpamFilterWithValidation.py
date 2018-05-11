# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:08:44 2018

@author: Andrew
"""



from sklearn.naive_bayes import GaussianNB
import sklearn.model_selection as skl
import numpy as np
from numpy import genfromtxt



# Using Naive Base to classify either Spam or Ham Then validating and reporting the 5-fold cross val error

# Getting data organized
my_data = genfromtxt("C:\\Users\\Elena Burger\\Downloads\\spambase.csv", delimiter=',') #get data

X_Data = my_data[:, :-1].copy() 
y_label = my_data[:,-1].copy()


gnb = GaussianNB()
output = skl.cross_val_score(gnb, X_Data, y= y_label, cv=5)
mean = np.mean(output)

print(mean)







