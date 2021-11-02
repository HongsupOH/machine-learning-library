import numpy as np
import matplotlib.pyplot as plt
import os
import sys

np.set_printoptions(threshold=sys.maxsize)
sys.path.append('../../PerceptronLearning')

from perceptron_standard import *

Data_train = np.loadtxt('../bank-note/train.csv',delimiter=',',dtype=np.double)
Data_test = np.loadtxt('../bank-note/test.csv',delimiter=',',dtype=np.double)

X_train,Y_train = Data_train[:,:-1],Data_train[:,-1]
one_train = np.ones((len(X_train),1))
X_train = np.hstack((one_train,X_train))

X_test,Y_test = Data_test[:,:-1],Data_test[:,-1]
one_test = np.ones((len(X_test),1))
X_test = np.hstack((one_test,X_test))

print("####### START Standard Perceptron #######")
r = 0.2
epoch = 10
errors = []
for trial in range(1):
    w = perceptron_standard(X_train,Y_train,r,epoch)
    y_pred_train = prediction_standard(w,X_train)
    y_pred_test = prediction_standard(w,X_test)
    Y_test = modify_label(Y_test)
    print("### START Trial = {} ###".format(trial+1))
    print("w = {}".format(w))
    print("test error = {}".format(100*len(Y_test[Y_test!=y_pred_test])/len(Y_test)))
    print("### END Trial = {} ###".format(trial+1))
    np.savetxt('Result_Q2_2_a/w_Trial={}.csv'.format(trial),w)
    np.savetxt('Result_Q2_2_a/test_error_Trial={}.csv'.format(trial),np.array([100*len(Y_test[Y_test!=y_pred_test])/len(Y_test)]))
    errors.append(100*len(Y_test[Y_test!=y_pred_test])/len(Y_test))

errors = np.array(errors)
print("Average error is {}%".format(np.mean(errors)))

print("####### END Standard Perceptron #######")
