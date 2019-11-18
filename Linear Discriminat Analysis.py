# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:43:08 2018

@author: Adila
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv('E:/Maestrado/Dataset/fars_data.csv', sep= ';', header= None)

#Divido los datos del trining y test
X = df.values[:, 0:28]
Y = df.values[:,29]

#defino los % del tamaño
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)



lda = LinearDiscriminantAnalysis()
model = lda.fit(X_train, y_train)

print(model.priors_)

print(model.means_)

print(model.coef_)

pred=model.predict(X_test)
print(np.unique(pred, return_counts=True))

#Matrizde confusion

print(confusion_matrix(pred, y_test))
print(classification_report(y_test, pred, digits=3))
