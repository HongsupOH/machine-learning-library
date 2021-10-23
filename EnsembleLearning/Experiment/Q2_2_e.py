import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../../DecisionTree')
sys.path.append('../../EnsembleLearning')

from IG3 import *
from helper import *
from bagging import *
from bv_decomposition import *


S_train,label_train = open_file('../bank-1/train.csv')
S_test,label_test = open_file('../bank-1/test.csv')


varDict = {'age':numericalAttribute(S_train,S_test,S_train[:,0],0),
          'job':["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",\
                 "blue-collar","self-employed","retired","technician","services"],
          'marital':["married","divorced","single"],
          'education':["unknown","secondary","primary","tertiary"],
          'default':["yes","no"],
          'balance':numericalAttribute(S_train,S_test,S_train[:,5],5),
          'housing':["yes","no"],
          'loan':["yes","no"],
          'contact':["unknown","telephone","cellular"],
          'day':numericalAttribute(S_train,S_test,S_train[:,9],9),
          'month':['dec','feb','sep','jun','mar','jul','may','aug','jan','oct','apr','nov'],
          'duration':numericalAttribute(S_train,S_test,S_train[:,11],11),
          'campaign':numericalAttribute(S_train,S_test,S_train[:,12],12),
          'pdays':numericalAttribute(S_train,S_test,S_train[:,13],13),
          'previous':numericalAttribute(S_train,S_test,S_train[:,14],14),
          'poutcome':["unknown","other","failure","success"]}

attributes = list(varDict)
    
num_sample = 1000
t = 500
iteration = 100
file = open('Result_Bias_Variance_subset=True.txt',"w")
print("START: Bias Variance Decomposition - Bagging")
for num_subset in [2,4,6]:
    single_bias, single_variance,single_square, ensemble_bias, ensemble_variance, ensemble_square =\
                 bias_variance_decomposition(S_train,S_test,label_test,varDict,attributes,iteration,num_sample,t,att_subsets = num_subset)
    file.write("Result of subset {}\n".format(num_subset))
    file.write("Single Tree\n")
    file.write("Bias :{}\n".format(single_bias))
    file.write("Variance :{}\n".format(single_variance))
    file.write("Square :{}\n".format(single_square))
    
    file.write("Whole Tree")
    file.write("Bias :{}\n".format(ensemble_bias))
    file.write("Variance :{}\n".format(ensemble_variance))
    file.write("Square :{}\n".format(ensemble_square))
print("END: Bias Variance Decomposition - Bagging")
