import numpy as np
import matplotlib.pyplot as plt
import os
import sys

np.set_printoptions(threshold=sys.maxsize)
sys.path.append('../../SVM')

from dual_problem import*
from helper import *

X_train,Y_train = gen_train_test('../bank-note/train.csv',none_one=True)
Y_train = modify_label(Y_train)
X_test,Y_test = gen_train_test('../bank-note/test.csv',none_one=True)
Y_test = modify_label(Y_test)
g = 0

for C in [100/873,500/873,700/873]:
    for g in [0.1,0.5,1,5,100]:
        print('START C = {}, g = {} ...'.format(C,g))

        alpha = np.zeros(X_train.shape[0])
        args = [Kernel,g]
        obj_fn = lambda alpha,args: objectiveFN(X_train,Y_train,alpha,args)
        alpha = optimization(obj_fn,Kernel,alpha,g,C,Y_train)
        alpha[np.isclose(alpha,0)] = 0
        alpha[np.isclose(alpha,C)] = C
        
        print("The number of support vector: {}".format(len(alpha[alpha>0])))

        b_opt = dual_bias(alpha,X_train,Y_train,0)
        y_hat = dual_prediction_kernel(alpha,X_train,X_train,Y_train,b_opt,g)
        y_hat_t = dual_prediction_kernel(alpha,X_train,X_test,Y_train,b_opt,g)
        err = len(Y_train[Y_train!=y_hat])/len(Y_train)
        err_t = len(Y_test[Y_test!=y_hat_t])/len(Y_test)
        
        print("Training error is {} %".format(err*100))
        print("Test error is {} %".format(err_t*100))
        print('END C = {}, g = {} ...\n'.format(C,g))

