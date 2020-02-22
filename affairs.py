# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:17:50 2019

@author: Ganesh
"""

import pandas as pd

affairsdata = pd.read_csv("F:\\R\\files\\affairs.csv")
#EDA process
affairsdata.columns
affairsdata = affairsdata.drop(["Unnamed: 0"], axis = 1)
affairsdata.affairs.value_counts()

#make two catagorics in dependent variable


affairsdata.affairs = affairsdata.affairs.replace(to_replace = [1,2,3,7,12], value = 1)

affairsdata = pd.get_dummies(affairsdata, columns = ["gender", "children"])

#na's
affairsdata.isna().sum()
affairsdata.isnull().sum() # this dataset has no na's and null values

#visualizations

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(affairsdata)
sns.countplot(affairsdata.affairs)

sns.distplot(affairsdata.age)

sns.pairplot(affairsdata)

#spliting the data into train and test

x = affairsdata.drop('affairs', axis = 1)
y = affairsdata['affairs']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression().fit(x_train,y_train)
predict = model1.predict(x_test)

predict

from sklearn.metrics import classification_report

classification_report(y_test, predict)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predict)
(128+5)/(128+3+2+45) #accuracy is 73.42 percentage

#building a model after normalizing 

from sklearn import preprocessing

ndata  = affairsdata

ndata.iloc[:,1:12] = preprocessing.normalize(ndata.iloc[:,1:12])

#split the data into test and train

a = ndata.drop('affairs', axis = 1)
b = ndata['affairs']

a_train, a_test, b_train, b_test = train_test_split(a,b,test_size = 0.3)

model2 = LogisticRegression().fit(a_train,b_train)

predictnorm = model2.predict(b_test)
classification_report(b_test, predict)

confusion_matrix(b_test, predict) #accuracy is 73.33 percentage

#normalizing the data doesnt affect the any accuracy of the testing datasets