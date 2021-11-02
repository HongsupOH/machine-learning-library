import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../../DecisionTree')

import matplotlib.pyplot as plt
from tabulate import tabulate
from IG3 import *
from entropy import entropy
from majority_error import majority_error
from gini import gini
from helper import *

S_train,label_train = open_file('../bank/train.csv')
S_test,label_test = open_file('../bank/test.csv')


varDict = {'age':numericalAttribute(S_train,S_test,S_train[:,0],0),
          'job':["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",\
                 "blue-collar","self-employed","retired","technician","services"],
          'marital':["married","divorced","single"],
          'education':["unknown","secondary","primary","tertiary"],
          'default':["yes","no"],
          'balance':numericalAttribute(S_train,S_test,S_train[:,5],5),
          'housing':["yes","no"],
          'loan':["yes","no"],
          'contact':["unknown","telephone","cellular"],
          'day':numericalAttribute(S_train,S_test,S_train[:,9],9),
          'month':['dec','feb','sep','jun','mar','jul','may','aug','jan','oct','apr','nov'],
          'duration':numericalAttribute(S_train,S_test,S_train[:,11],11),
          'campaign':numericalAttribute(S_train,S_test,S_train[:,12],12),
          'pdays':numericalAttribute(S_train,S_test,S_train[:,13],13),
          'previous':numericalAttribute(S_train,S_test,S_train[:,14],14),
          'poutcome':["unknown","other","failure","success"]}
attributes = list(varDict)
print("###### Q2.3 START ######")
print("### Q2.3 (a) START ###")
print('1. Entropy')
train_err = np.zeros((16,3))
test_err = np.zeros((16,3))

fn = entropy
errsTr1 = []
errsTe1 = []

for maxDepth in range(1,17):
    cur_depth = 1
    parent,edge = None,None
    Tree1 = Tree(maxDepth)
    
    IG3(Tree1,S_train,label_train,attributes,varDict,cur_depth,fn,parent,edge)
    
    y_pred_train1 = prediction(Tree1,S_train,varDict)
    y_pred_test1 = prediction(Tree1,S_test,varDict)
    errsTr1.append(error(label_train,y_pred_train1))
    errsTe1.append(error(label_test,y_pred_test1))

train_err[:,0] = np.array(errsTr1)
test_err[:,0] = np.array(errsTe1)

print('2. ME')
errsTr2 = []
errsTe2 = []

fn = majority_error
for maxDepth in range(1,17):
    cur_depth = 1
    parent,edge = None,None
    Tree2 = Tree(maxDepth)
    
    IG3(Tree2,S_train,label_train,attributes,varDict,cur_depth,fn,parent,edge)
    
    y_pred_train2 = prediction(Tree2,S_train,varDict)
    y_pred_test2 = prediction(Tree2,S_test,varDict)
    errsTr2.append(error(label_train,y_pred_train2))
    errsTe2.append(error(label_test,y_pred_test2))

train_err[:,1] = np.array(errsTr2)
test_err[:,1] = np.array(errsTe2) 

print('3. GINI')
fn = gini
errsTr3 = []
errsTe3 = []
data3 = []
for maxDepth in range(1,17):
    cur_depth = 1
    parent,edge = None,None
    Tree3 = Tree(maxDepth)
    
    IG3(Tree3,S_train,label_train,attributes,varDict,cur_depth,fn,parent,edge)
    
    y_pred_train3 = prediction(Tree3,S_train,varDict)
    y_pred_test3 = prediction(Tree3,S_test,varDict)
    errsTr3.append(error(label_train,y_pred_train3))
    errsTe3.append(error(label_test,y_pred_test3))

train_err[:,2] = np.array(errsTr2)
test_err[:,2] = np.array(errsTe2)

print('4. Average prediction')
print('4. Average prediction for Training data')
print(tabulate(train_err,headers=['Entropy','Majority Error','GINI']))
print()
print('4. Average prediction for Test data')
print(tabulate(test_err,headers=['Entropy','Majority Error','GINI']))

print("### Q2.3 (a) END ###")
print("### Q2.3 (b) START ###")
S_train_fix = missingData(S_train)
S_test_fix = missingData(S_test)
train_err = np.zeros((16,3))
test_err = np.zeros((16,3))
print('1. Entropy')
fn = entropy
errsTr1 = []
errsTe1 = []
for maxDepth in range(1,17):
    cur_depth = 1
    parent,edge = None,None
    Tree1 = Tree(maxDepth)
    
    IG3(Tree1,S_train,label_train,attributes,varDict,cur_depth,fn,parent,edge)
    
    y_pred_train1 = prediction(Tree1,S_train,varDict)
    y_pred_test1 = prediction(Tree1,S_test,varDict)
    errsTr1.append(error(label_train,y_pred_train1))
    errsTe1.append(error(label_test,y_pred_test1))
    
    
train_err[:,0] = np.array(errsTr1)
test_err[:,0] = np.array(errsTe1)


print('2. ME')
errsTr2 = []
errsTe2 = []
fn = majority_error
for maxDepth in range(1,17):
    cur_depth = 1
    parent,edge = None,None
    Tree2 = Tree(maxDepth)
    
    IG3(Tree2,S_train,label_train,attributes,varDict,cur_depth,fn,parent,edge)
    
    y_pred_train2 = prediction(Tree2,S_train,varDict)
    y_pred_test2 = prediction(Tree2,S_test,varDict)
    errsTr2.append(error(label_train,y_pred_train2))
    errsTe2.append(error(label_test,y_pred_test2))
    
    
train_err[:,1] = np.array(errsTr2)
test_err[:,1] = np.array(errsTe2)


print('3. GINI')
fn = gini
errsTr3 = []
errsTe3 = []
data3 = []
for maxDepth in range(1,17):
    cur_depth = 1
    parent,edge = None,None
    Tree3 = Tree(maxDepth)
    
    IG3(Tree3,S_train,label_train,attributes,varDict,cur_depth,fn,parent,edge)
    
    y_pred_train3 = prediction(Tree3,S_train,varDict)
    y_pred_test3 = prediction(Tree3,S_test,varDict)
    errsTr3.append(error(label_train,y_pred_train3))
    errsTe3.append(error(label_test,y_pred_test3))

train_err[:,2] = np.array(errsTr2)
test_err[:,2] = np.array(errsTe2)

print('4. Average prediction for Train data')
print(tabulate(train_err,headers=['Entropy','Majority Error','GINI']))
print()
print('4. Average prediction for Test data')
print(tabulate(test_err,headers=['Entropy','Majority Error','GINI']))
print("### Q2.3 (b) END ###")
print("###### Q2.3 END ######")

