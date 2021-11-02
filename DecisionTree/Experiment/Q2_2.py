import numpy as np
import os
import sys

sys.path.append('../../DecisionTree')

from tabulate import tabulate
from IG3 import *
from entropy import entropy
from majority_error import majority_error
from gini import gini
from helper import *


S_train,label_train = open_file('../car/train.csv')
S_test,label_test = open_file('../car/test.csv')

varDict = {'buying':['vhigh', 'high', 'med', 'low'],
          'maint':['vhigh', 'high', 'med', 'low'],
          'doors':['2', '3', '4', '5more'],
          'persons':['2', '4', 'more'],
          'lug_boot':['small', 'med', 'big'],
          'safety':['low', 'med', 'high']}

attributes = list(varDict)

print("###### Q2.2 START ######")
print('1. Entropy')
fn = entropy

train_err = np.zeros((6,3))
test_err = np.zeros((6,3))

errsTr1 = []
errsTe1 = []

for maxDepth in range(1,7):
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


#print(tabulate(data1,headers=['Depth','Train','Test']))

print('2. Majority Error')
fn = majority_error
errsTr2 = []
errsTe2 = []
for maxDepth in range(1,7):
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


#print(tabulate(data2,headers=['Depth','Train','Test']))

print('3. GINI')
fn = gini
errsTr3 = []
errsTe3 = []

for maxDepth in range(1,7):
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
    

#print(tabulate(data3,headers=['Depth','Train','Test']))

print('4. Average prediction for Training data')
print(tabulate(train_err,headers=['Entropy','Majority Error','GINI']))
print()
print('4. Average prediction for Test data')
print(tabulate(test_err,headers=['Entropy','Majority Error','GINI']))

print("###### Q2.2 END ######")
