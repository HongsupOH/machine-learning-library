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
errsTr1 = []
errsTe1 = []
data1 = []
for maxDepth in range(1,7):
    cur_depth = 1
    parent,edge = None,None
    Tree1 = Tree(maxDepth)
    
    IG3(Tree1,S_train,label_train,attributes,varDict,cur_depth,fn,parent,edge)
    
    y_pred_train1 = prediction(Tree1,S_train,varDict)
    y_pred_test1 = prediction(Tree1,S_test,varDict)
    errsTr1.append(error(label_train,y_pred_train1))
    errsTe1.append(error(label_test,y_pred_test1))
    data1.append([maxDepth, errsTr1[-1],errsTe1[-1]])
print(tabulate(data1,headers=['Depth','Train','Test']))

print('2. Majority Error')
fn = majority_error
errsTr2 = []
errsTe2 = []
data2 = []
for maxDepth in range(1,7):
    cur_depth = 1
    parent,edge = None,None
    Tree2 = Tree(maxDepth)
    
    IG3(Tree2,S_train,label_train,attributes,varDict,cur_depth,fn,parent,edge)
    
    y_pred_train2 = prediction(Tree2,S_train,varDict)
    y_pred_test2 = prediction(Tree2,S_test,varDict)
    errsTr2.append(error(label_train,y_pred_train2))
    errsTe2.append(error(label_test,y_pred_test2))
    data2.append([maxDepth, errsTr2[-1],errsTe2[-1]])
print(tabulate(data2,headers=['Depth','Train','Test']))

print('3. GINI')
fn = gini
errsTr3 = []
errsTe3 = []
data3 = []
for maxDepth in range(1,7):
    cur_depth = 1
    parent,edge = None,None
    Tree3 = Tree(maxDepth)
    
    IG3(Tree3,S_train,label_train,attributes,varDict,cur_depth,fn,parent,edge)
    
    y_pred_train3 = prediction(Tree3,S_train,varDict)
    y_pred_test3 = prediction(Tree3,S_test,varDict)
    errsTr3.append(error(label_train,y_pred_train3))
    errsTe3.append(error(label_test,y_pred_test3))
    data3.append([maxDepth, errsTr3[-1],errsTe3[-1]])

print(tabulate(data3,headers=['Depth','Train','Test']))

print('4. Average prediction')
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.array(data3)

avg_data = []
en_avg = ['Entropy']+np.mean(data1[:,1:],axis=0).tolist()
avg_data.append(en_avg)
me_avg = ['ME']+np.mean(data2[:,1:],axis=0).tolist()
avg_data.append(me_avg)
gini_avg = ['GINI']+np.mean(data3[:,1:],axis=0).tolist()
avg_data.append(gini_avg)
print(tabulate(avg_data,headers=['Method','Train','Test']))
print("###### Q2.2 END ######")
