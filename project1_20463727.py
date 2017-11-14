#!usr/bin/env python
#-*- coding: utf-8 -*-
 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import sklearn
from sklearn import preprocessing
from sklearn import cross_validation,svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB  

x_train = pd.read_csv(r'C:/Users/wangyicheng/Desktop/6000Bproject/project1/traindata.csv')
y_train = pd.read_csv(r'C:/Users/wangyicheng/Desktop/6000Bproject/project1/trainlabel.csv')
x_test = pd.read_csv(r'C:/Users/wangyicheng/Desktop/6000Bproject/project1/testdata.csv')

#data prprocessing
train_scaled = preprocessing.scale(x_train)
test_scaled = preprocessing.scale(x_test) 
train_scaled =  pd.DataFrame(data = train_scaled)


y_train = np.ravel(y_train)


# clf = svm.SVR()  
# clf = BernoulliNB()
# clf = GaussianNB().fit(train_scaled,y_train) 
clf = SVC(probability=True, kernel='rbf')
clf.fit(train_scaled,y_train)
scores = cross_validation.cross_val_score(clf, train_scaled, y_train, cv = 5)
# # joblib.dump(clf,'svm_model1.pkl')
print(scores)
# clf = SVC(probability=True, kernel='rbf')
# clf.fit(train_scaled,y_train)
 
predictions = clf.predict(test_scaled)
np.savetxt("C:/Users/wangyicheng/Desktop/testdatay.csv", predictions, delimiter=",")
print(predictions)
