# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
""" 
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

bank = pd.read_csv("F:\\R\\files\\bank-full.csv", sep = ';')

#just convert the catogorical variable to the numerical datas.
#job, marital, education, default, housing, loan, contact, month, poutcome

bank.columns
bank = pd.get_dummies(bank, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
# other columns age, balance, day, duration, campaign, pdays, previous

bank.y.value_counts()

#find missing values
bank.isna().sum()

#normalize the data

from sklearn import preprocessing

bank.iloc[:,0:7] = preprocessing.normalize(bank.iloc[:,0:7])

#split the data into train and test

from sklearn.model_selection import train_test_split

x = bank.drop('y', axis = 1)
y = bank['y']

from sklearn.linear_model import LogisticRegression
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)

import statsmodels.formula.api as sms
train_model1 = sms.logit("y ~ age + balance + day + duration + campaign + pdays + previous + job_admin. + job_blue-collar +job_entrepreneur + job_housemaid + job_management + job_retired + job_self-employed + job_services + job_student + job_technician + job_unemployed + job_unknown + marital_divorced + marital_married + marital_single + education_primary + education_secondary + education_tertiary + education_unknown + default_no + default_yes + housing_no + housing_yes + loan_no + loan_yes + contact_cellular + contact_telephone + contact_unknown + month_apr + month_aug + month_dec + month_feb + month_jan + month_jul + month_jun + month_mar + month_may", data = x_train, class = y_train).fit()

train_model1 = sms.logit("y_train~x_train").fit()

model1 = LogisticRegression().fit(x_train, y_train)

?sms.logit
model1.coef_
predict = model1.predict(y_test)

from sklearn.metrics import classification_report

classification_report(y_test, predict)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predict)
