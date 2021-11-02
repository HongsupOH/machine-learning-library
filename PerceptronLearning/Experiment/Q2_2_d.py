import numpy as np
import pandas as pd
import os
import sys

sys.path.append('../../PerceptronLearning')

from perceptron_standard import *
from perceptron_vote import *
from perceptron_average import *

Data_train = np.loadtxt('../bank-note/train.csv',delimiter=',',dtype=np.double)
Data_test = np.loadtxt('../bank-note/test.csv',delimiter=',',dtype=np.double)
X_train,Y_train = Data_train[:,:-1],Data_train[:,-1]
one_train = np.ones((len(X_train),1))
X_train = np.hstack((one_train,X_train))

X_test,Y_test = Data_test[:,:-1],Data_test[:,-1]
one_test = np.ones((len(X_test),1))
X_test = np.hstack((one_test,X_test))

r = 0.2
data = []
epoch = 10
errs_st,errs_vote,errs_average =[],[],[]
print("####### START Compare three methods #######")
for trial in range(50):
    row = []

    w = perceptron_standard(X_train,Y_train,r,epoch)
    
    y_pred_test = prediction_standard(w,X_test)
    Y_test = modify_label(Y_test)

    err_st = len(Y_test[Y_test!=y_pred_test])/len(Y_test)
    errs_st.append(err_st)
    row.append(err_st*100)
    W,C = perceptron_vote(X_train,Y_train,r,epoch)
    
    y_pred_test = prediction_vote(W,C,X_test)
    Y_test = modify_label(Y_test)

    err_vote = len(Y_test[Y_test!=y_pred_test])/len(Y_test)
    errs_vote.append(err_vote)
    row.append(err_vote*100)

    a = perceptron_average(X_train,Y_train,r,epoch)
    
    y_pred_test = prediction_average(a,X_test)
    Y_test = modify_label(Y_test)

    err_average = len(Y_test[Y_test!=y_pred_test])/len(Y_test)
    errs_average.append(err_average)
    row.append(err_average*100)

    data.append(row)

data = np.array(data)

errs_st = np.array(errs_st)
errs_vote = np.array(errs_vote)
errs_average = np.array(errs_average)

dt = np.array([[np.mean(errs_st)*100,np.mean(errs_vote)*100,np.mean(errs_average)*100]])

ind = ['Average']
labels = ["Standard(%)","Vote(%)","Average(%)"]
df = pd.DataFrame(dt, index=ind ,columns = labels)
print(df)
df.to_csv('Result_Q2_2_d/Errors.csv')
print("####### END Compare three methods #######")






    
