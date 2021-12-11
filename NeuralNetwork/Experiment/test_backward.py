import numpy as np
import matplotlib.pyplot as plt
import os
import sys

np.set_printoptions(threshold=sys.maxsize)
sys.path.append('../../NeuralNetwork')

from NeuralNetwork import*
from helper import *

X = np.array([[1,1,1]])
Y = np.array([1])

NN = NeuralNetwork(X,Y)

NN.add_layer(2,NN.sigmoid, NN.gaussian_standard_weight)
weight = np.array([[-1,1],
                   [-2,2],
                   [-3,3]])

NN.update_weight(weight,0)

NN.add_layer(2,NN.sigmoid, NN.gaussian_standard_weight)
weight = np.array([[-1,1],
                   [-2,2],
                   [-3,3]])

NN.update_weight(weight,1)


NN.add_layer(1,NN.linear, NN.gaussian_standard_weight)
weight = np.array([[-1],
                   [2],
                   [-1.5]])

NN.update_weight(weight,2)

y = NN.forward()
print("***Check 1-2 Forward path ***")
for layer in NN.structure:
    z = NN.structure[layer].z
    print("At Layer {}".format(layer))
    print("z = {}".format(z))
print("***End Forward path***\n")

NN.backward(y,np.array([1]))
print("***Check 1-3 Backward path ***")
layers = list(NN.structure)[::-1]
for layer in layers:
    dLdW = NN.structure[layer].der_weight
    print("At Layer {}".format(layer))
    print("dL/dW = {}".format(dLdW))
print("***End Backward path***\n")

    

