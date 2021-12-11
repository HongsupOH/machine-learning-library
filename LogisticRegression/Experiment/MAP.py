import numpy as np
import matplotlib.pyplot as plt
import os
import sys

np.set_printoptions(threshold=sys.maxsize)
sys.path.append('../../LogisticRegression')

from logistic_regression import *
from helper import *

X_train,Y_train = gen_train_test('../bank-note/train.csv')
Y_train = modify_label(Y_train)
X_test,Y_test = gen_train_test('../bank-note/test.csv')
Y_test = modify_label(Y_test)

epoch = 100
r0s = [0.01,0.0025, 0.00125, 0.000625]
a_s = [1,0.5,0.1, 0.05, 0.01, 0.001,0.0005]
mode = 1
print("##### Q3-(a) MAP START #####")
for v in [0.01,0.1,0.5,1,3,5,10,100]:
    err_opt = np.inf
    for r0 in r0s:
        for a in a_s:
            w = logistic_regression_MAP(X_train,Y_train,epoch,r0,a,v,mode)
            Y_hat = prediction(w,X_train)
            err = len(Y_train[Y_train!=Y_hat])/len(Y_train)
            if err < err_opt:
                err_opt = err
                r0_opt,a_opt = r0,a
                w_opt = w
    print('***Prior var: {}***'.format(v))
    print('Best r0={},a={}'.format(r0_opt,a_opt))
    print('Training Error = {}%'.format(err_opt*100))
    Y_hat_t = prediction(w_opt,X_test)
    err = len(Y_test[Y_test!=Y_hat_t])/len(Y_test)
    print('Test Error = {}%'.format(err*100))
print("##### Q3-(a) MAP END #####")
