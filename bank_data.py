# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:18:16 2019

@author: Good Guys
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm
import seaborn as sns
sns.set()



bd = pd.read_csv('C:\\Users\\Good Guys\\Desktop\\pRACTICE\\EXCELR PYTHON\\Assignment\\Logistic Regression\\bank_data (1).csv')
bd.columns
len(bd.columns)
bd.head()
bd.describe(include = 'all')

X = bd.iloc[:,0:30].values
Y = bd.iloc[:,[31]].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

x1 = sm.add_constant(X)
reg_log = sm.Logit(Y,x1)
results = reg_log.fit()
results.summary()

#Predict
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test, y_pred)
print(cn)
per = cn[0,0] + cn[1,1]
p = per + cn[0,1] + cn[1,0]
per / p

#ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
roc_auc = roc_auc_score(Y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label = 'Logistic Regression Sensitivity = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.title('ROC')
plt.legend(loc="lower Right")
plt.show()