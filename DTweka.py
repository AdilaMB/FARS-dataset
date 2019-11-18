# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
#from sklearn.tree import export_graphviz
#from graphviz
#from mathplotlib.pyplot as plt 
import weka.core.converters.ConverterUtils.DataSource as DS
import weka.filters.Filter as Filter
import os



df = DS.read_csv('E:/Maestrado/Dataset/fars_data.csv', sep= ';', header= None)

#Divido los datos del trining y test
X = df.values[:, 0:28]
Y = df.values[:,29]

#defino los % del tamaño
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#Profundidad 3, número mínimo de muestras requeridas para estar en un nodo hoja = 5.
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

y_pred_en = clf_entropy.predict(X_test)

acc = accuracy_score(y_test,y_pred)*100

print(y_pred)
print(y_pred_en)

print(acc)

#Pintando el arbol
## exportar el modelo a archivo .dot
#with open(r"tree1.dot", 'w') as f:
#     f = tree.export_graphviz(decision_tree,
#                             out_file=f,
#                              max_depth = 7,
#                             impurity = True,
#                              rounded = True,
#                              filled= True )
        
# Convertir el archivo .dot a png para poder visualizarlo
#check_call(['dot','-Tpng',r'tree1.dot','-o',r'tree1.png'])
#PImage("tree1.png")

