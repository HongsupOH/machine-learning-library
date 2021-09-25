import numpy as np
import os
from IG3 import *
from entropy import entropy
from majority_error import majority_error
from gini import gini
from helper import *
from tabulate import tabulate

path = "car"
os.chdir(path)

S_train,label_train = open_file('train.csv')
S_test,label_test = open_file('test.csv')

v_dict = {'buying':['vhigh', 'high', 'med', 'low'],
          'maint':['vhigh', 'high', 'med', 'low'],
          'doors':['2', '3', '4', '5more'],
          'persons':['2', '4', 'more'],
          'lug_boot':['small', 'med', 'big'],
          'safety':['low', 'med', 'high']}


print("###### Q2.2 START ######")
print('1. Entropy')
fn = entropy
errsTr1 = []
errsTe1 = []
data1 = []
for maxDepth in range(1,7):
    Tree1 = genTree(S_train,v_dict,label_train,maxDepth,fn)
    y_pred_train1 = predict(Tree1,S_train,label_train,v_dict)
    y_pred_test1 = predict(Tree1,S_test,label_test,v_dict)
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
    Tree2 = genTree(S_train,v_dict,label_train,maxDepth,fn)
    y_pred_train2 = predict(Tree2,S_train,label_train,v_dict)
    y_pred_test2 = predict(Tree2,S_test,label_test,v_dict)
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
    Tree3 = genTree(S_train,v_dict,label_train,maxDepth,fn)
    y_pred_train3 = predict(Tree3,S_train,label_train,v_dict)
    y_pred_test3 = predict(Tree3,S_test,label_test,v_dict)
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
