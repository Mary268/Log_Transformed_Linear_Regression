#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 22:06:08 2018

Author: Menglu Wang
ID:     20707728

@author: marywang
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz


from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
import sklearn.ensemble as ensemble
from sklearn import svm
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pulp


filename = "2.BLINK.HAREMS.csv"
filename2 = "demand_variables.csv"
blink_data = pd.read_csv(filename, dtype='object', encoding='utf-8', engine='c')
blink_data = blink_data.apply(pd.to_numeric, errors='ignore')
blink_variables = pd.read_csv(filename2, dtype='object', encoding='utf-8', engine='c')
blink_response =blink_variables["response"]
blink_variables.drop('response', axis=1, inplace=True)

#pd.DataFrame(blink_data['sales_unit'],blink_data['discount_per'],
#                               blink_data['discount_per'],blink_data['discount_per']
#                               blink_data['discount_per'])

#---------------------------Simple Decision Tree Classifier--------------------
clf = tree.DecisionTreeClassifier()

clf = clf.fit(blink_variables, blink_response)
#clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("blink")

#---------------------------Simple Decision Tree Regressor---------------------
regressor = tree.DecisionTreeRegressor(max_depth=4,max_features=3,min_samples_leaf=15,min_samples_split=2)
regressor = regressor.fit(blink_variables, blink_response)
dot_data = tree.export_graphviz(regressor, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("regressor")

#---------------------------Random Forest Decision Tree Classifier--------------
forest_clf = ensemble.RandomForestClassifier(n_estimators=20, min_samples_split=2, max_features = 5, 
                                                  max_depth = 5, min_samples_leaf = 15)
forest_clf = forest_clf.fit(blink_variables, blink_response)
#---------------------------Random Forest Decision Tree Regressor--------------
forest_regressor = ensemble.RandomForestRegressor(n_estimators=20, min_samples_split=2, max_features = 3, 
                                                  max_depth = 4, min_samples_leaf = 15)
forest_regressor = forest_regressor.fit(blink_variables, blink_response)

#---------------------------AdaBoost Decision Tree Classifier------------------
ada_classifier = ensemble.AdaBoostClassifier()
ada_classifier.fit(blink_variables, blink_response)

#---------------------------AdaBoost Decision Tree Regressor-------------------
ada_regressor = ensemble.AdaBoostRegressor(n_estimators=10000, random_state = 42)
ada_regressor.fit(blink_variables, blink_response)

#---------------------------Support Vector Machine-----------------------------
svm = svm.SVC()
svm.fit(blink_variables, blink_response)

#---------------------------Support Vector Machine-----------------------------
LR = LinearRegression()
LR.fit(blink_variables, blink_response)

#---------------------------Scores of each Models-------------------
blink_testdata = pd.read_csv("demand_testdata.csv", dtype='object', encoding='utf-8', engine='c')
blink_testdata = blink_testdata.apply(pd.to_numeric, errors='ignore')
blink_testresponse = blink_testdata["response"]
blink_testdata.drop('response', axis=1, inplace=True)
#R^2 of the prediction:
for clf_item, label in zip([clf, regressor, forest_clf, forest_regressor, 
                            ada_classifier, ada_regressor, svm, LR], 
                      ['DecisionTree Classifier', 'DecisionTree Regressor',
                       'Random Forest Classifier', 'Random Forest Regresssor',
                       'AdaBoost Forest Classifier', 'AdaBoost Forest Regressor',
                       'Support Vector Machine', 'Linear Regression']):
    print("                         ")
    print(clf_item.score(blink_variables, blink_response), label + "  R Squared ")
    #scores = cross_val_score(clf_item, blink_variables, blink_response)
    #print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    #print(clf_item.score(blink_variables[18:24], blink_response[18:24]), label)
    #print(clf_item.score(blink_variables[43:47], blink_response[43:47]), label)
    print(clf_item.score(blink_testdata[0:4], blink_testresponse[0:4]), label+ "  Validation of Next 4 Weeks ")

#---------------------------EOSS Predictions of Models-------------------
clf_pred = clf.predict(blink_testdata).astype(int)
clf_pred_error = clf_pred - blink_testresponse

regressor_pred = regressor.predict(blink_testdata).astype(int)
regressor_pred_error = regressor_pred - blink_testresponse

forest_clf_pred = forest_clf.predict(blink_testdata).astype(int)
forest_clf_pred_error = forest_clf_pred - blink_testresponse

forest_regressor_pred = forest_regressor.predict(blink_testdata).astype(int)
forest_regressor_pred_error = forest_regressor_pred - blink_testresponse

ada_classifier_pred = ada_classifier.predict(blink_testdata).astype(int)
ada_classifier_pred_error = ada_classifier_pred - blink_testresponse

ada_regressor_pred = ada_regressor.predict(blink_testdata).astype(int)
ada_regressor_pred_error = ada_regressor_pred - blink_testresponse

svm_pred = svm.predict(blink_testdata).astype(int)
svm_pred_error = svm_pred - blink_testresponse

LR_pred = LR.predict(blink_testdata).astype(int)
LR_pred_error = LR_pred - blink_testresponse

pred_error_EOSS = pd.DataFrame()
pred_error_EOSS['clf_pred_error'] = clf_pred_error[0:4]
pred_error_EOSS['regressor_pred_error'] = regressor_pred_error[0:4]
pred_error_EOSS['forest_clf_pred_error'] = forest_clf_pred_error[0:4]
pred_error_EOSS['forest_regressor_pred_error'] = forest_regressor_pred_error[0:4]
pred_error_EOSS['ada_classifier_pred_error'] = ada_classifier_pred_error[0:4]
pred_error_EOSS['ada_regressor_pred_error'] = ada_regressor_pred_error[0:4]
pred_error_EOSS['svm_pred_error'] = svm_pred_error[0:4]
pred_error_EOSS['LR_pred_error'] = LR_pred_error[0:4]

pred_error = pd.DataFrame()
pred_error['clf_pred_error'] = clf_pred_error
pred_error['regressor_pred_error'] = regressor_pred_error
pred_error['forest_clf_pred_error'] = forest_clf_pred_error
pred_error['forest_regressor_pred_error'] = forest_regressor_pred_error
pred_error['ada_classifier_pred_error'] = ada_classifier_pred_error
pred_error['ada_regressor_pred_error'] = ada_regressor_pred_error
pred_error['svm_pred_error'] = svm_pred_error
pred_error['LR_pred_error'] = LR_pred_error

print(pred_error_EOSS)

