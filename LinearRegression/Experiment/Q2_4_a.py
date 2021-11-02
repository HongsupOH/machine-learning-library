import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../../DecisionTree')
sys.path.append('../../LinearRegression')

from helper import *
from batch_gradient import *

S_train,label_train = open_file('../concrete/train.csv')
one_train = np.ones((len(S_train),1))
S_train =  np.array(S_train,dtype=float)
S_train = np.hstack((one_train,S_train))

label_train =  np.array(label_train,dtype=float)

S_test,label_test = open_file('../concrete/test.csv')
one_test = np.ones((len(S_test),1))
S_test =  np.array(S_test,dtype=float)
S_test = np.hstack((one_test,S_test))
label_test =  np.array(label_test,dtype=float)


T = 10000

r = 1
threshold = 1e-6
print("START Batch")
w,costs = batch_gradient(S_train,label_train,T,r,threshold)
w_opt = np.linalg.inv(S_train.T.dot(S_train)).dot(S_train.T.dot(label_train))
print("END Batch")
print("w = {}".format(w))
print("w_opt = {}".format(w_opt))

plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost (J)')
plt.grid()
plt.savefig('costs_batch.png')

plt.figure()
plt.plot(S_train.dot(w),label='batch')
plt.plot(label_train,label='true')
plt.grid()
plt.legend()
plt.savefig('result_batch.png')
