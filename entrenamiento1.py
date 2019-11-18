# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn import tree

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('E:/Maestrado/Dataset/fars_data.csv', sep= ';', header= None)

#Divido los datos del trining y test
X = df.values[:, 0:28]
Y = df.values[:,29]

#defino los % del tamaño
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#Naive Bayes-Gaussian

GausNB = GaussianNB()
GausNB.fit(X_train, y_train)
print (GausNB)

y_pred = GausNB.predict(X_test)

print (accuracy_score( y_test, y_pred))

#Naive Bayes-Bernouli

BernNB = BernoulliNB(binarize = True)
BernNB.fit(X_train, y_train)
print (BernNB)

y_expet = y_test
y_pred = BernNB.predict(X_test)

print (accuracy_score( y_test, y_pred))

#Naive Bayes-Multinomial

MultiNB = MultinomialNB()
MultiNB.fit(X_train, y_train)
print (MultiNB)

y_expet = y_test
y_pred = MultiNB.predict(X_test)

print (accuracy_score( y_test, y_pred))