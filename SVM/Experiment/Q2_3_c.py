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
gamma = [0.1,0.5,1,5,100]
alpha_dict = {}
print("START Q2-3-c")
for C in [500/873]:
    for ind,g in enumerate([0.1,0.5,1,5,100]):
        print('START C = {}, g = {} ...'.format(C,g))
        alpha = np.zeros(X_train.shape[0])
        args = [Kernel,g]
        obj_fn = lambda alpha,args: objectiveFN(X_train,Y_train,alpha,args)
        alpha = optimization(obj_fn,Kernel,alpha,g,C,Y_train)
        alpha[np.isclose(alpha,0)] = 0
        alpha[np.isclose(alpha,C)] = C
        print("The number of support vector: {}".format(len(alpha[alpha>0])))
        alpha_dict[ind] = alpha
        if ind!=0:
            print("Compare {} and {}".format(g,gamma[ind-1]))
            prev_alpha = alpha_dict[ind-1]
            same_len = np.sum(alpha[alpha>0]==prev_alpha[alpha>0])
            print("{} elements are same".format(same_len))
        print('END C = {}, g = {} ...\n'.format(C,g))
print("END Q2-3-c")
