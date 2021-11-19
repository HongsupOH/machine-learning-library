import numpy as np
import matplotlib.pyplot as plt
import os
import sys

np.set_printoptions(threshold=sys.maxsize)
sys.path.append('../../SVM')

from stochastic_sub_gradient_descent import*
from helper import *

X_train,Y_train = gen_train_test('../bank-note/train.csv')
Y_train = modify_label(Y_train)
X_test,Y_test = gen_train_test('../bank-note/test.csv')
Y_test = modify_label(Y_test)

epoch = 100

r0s = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
a_s = [1,0.5,0.1, 0.05, 0.01, 0.001,0.0005]
mode = 1
print('Mode 1')
for C in [100/873,500/873,700/873]:
    err_opt = np.inf
    for r0 in r0s:
        for a in a_s:
            w = stochastic_sub_gradient_descent(X_train,Y_train,C,r0,a,epoch,mode)
            Y_hat = prediction(w,X_train)
            err = len(Y_train[Y_train!=Y_hat])/len(Y_train)
            if err < err_opt:
                err_opt = err
                r0_opt,a_opt = r0,a
                w_opt = w
                
    print('Best r0={},a={}'.format(r0_opt,a_opt))
    print('Training Error = {}%'.format(err_opt*100))
    Y_hat_t = prediction(w_opt,X_test)
    err = len(Y_test[Y_test!=Y_hat_t])/len(Y_test)
    print('Test Error = {}%'.format(err*100))
    
print('Mode 2')   
mode = 2
for C in [100/873,500/873,700/873]:
    err_opt = np.inf
    for r0 in r0s:
        for a in a_s:
            w = stochastic_sub_gradient_descent(X_train,Y_train,C,r0,a,epoch,mode)
            Y_hat = prediction(w,X_train)
            err = len(Y_train[Y_train!=Y_hat])/len(Y_train)
            if err < err_opt:
                err_opt = err
                r0_opt,a_opt = r0,a
                w_opt = w
                
    print('Best r0={},a={}'.format(r0_opt,a_opt))
    print('Trainining Error = {}%'.format(err_opt*100))
    Y_hat_t = prediction(w_opt,X_test)
    err = len(Y_test[Y_test!=Y_hat_t])/len(Y_test)
    print('Test Error = {}%'.format(err*100))
    
    
