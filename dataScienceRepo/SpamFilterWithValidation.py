# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:08:44 2018

@author: Andrew
"""



from sklearn.naive_bayes import GaussianNB
import sklearn.model_selection as skl
import pandas as pd

gnb = GaussianNB()

# Using Naive Base to classify either Spam or Ham Then validating and reporting the 5-fold cross val error


Data = pd.read_csv("C:\\Users\\Public\\ECE523\\Data\\CSV\\spambase.csv")
numPyData = Data.values

X_Data = numPyData[:, :-1].copy()
y_label = numPyData[:,-1].copy()


output = skl.cross_val_score(gnb, X_Data, y= y_label)

print(output)







