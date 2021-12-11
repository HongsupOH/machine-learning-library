import numpy as np
import matplotlib.pyplot as plt
import os
import sys

np.set_printoptions(threshold=sys.maxsize)
sys.path.append('../../NeuralNetwork')

from NeuralNetwork import*
from stochastic_gradient_descent import stochastic_gradient_descent
from helper import *

X_train,Y_train = gen_train_test('../bank-note/train.csv')
Y_train = modify_label(Y_train)
X_test,Y_test = gen_train_test('../bank-note/test.csv')
Y_test = modify_label(Y_test)

epoch = 10
r0s = [0.01,0.5,0.0025, 0.00125]
a_s = [1,0.5,0.1, 0.05, 0.01, 0.001,0.0005]
mode = 1
print("##### Q2-(a) Gaussian distribution START #####")
for width  in [5,10,25,50,100]:
    err_opt = np.inf
    for r0 in r0s:
        for a in a_s:
            NN = NeuralNetwork(np.array([X_train[0]]),Y_train[0])
            NN.add_layer(width,NN.sigmoid, NN.gaussian_standard_weight)
            NN.add_layer(width,NN.sigmoid, NN.gaussian_standard_weight)
            NN.add_layer(1,NN.linear, NN.gaussian_standard_weight)
            Losses = stochastic_gradient_descent(X_train,Y_train,epoch,NN,r0,a,mode)
            predict = NN.prediction(X_train)
            predict_test = NN.prediction(X_test)
            err = len(Y_train[Y_train!=predict])/len(Y_train)
            if err < err_opt:
                err_opt = err
                r0_opt,a_opt = r0,a
                err_test_opt = len(Y_test[Y_test!=predict_test])/len(Y_test)*100
    print("***WIDTH - {}***".format(width))
    print('Best r0={},a={}'.format(r0_opt,a_opt))
    print('Training Error = {}%'.format(err_opt))
    print('Test Error = {}%'.format(err_test_opt))
print("##### Q2-(a) Gaussian distribution END #####")
            
