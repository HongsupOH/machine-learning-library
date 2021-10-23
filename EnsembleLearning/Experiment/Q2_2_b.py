import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging

sys.path.append('../../DecisionTree')
sys.path.append('../../EnsembleLearning')

from helper import *
from bagging import *


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

err_train = []
err_test = []
print("Q2-2-b: START bagging")
t = 500
num_sample = S_train.shape[0]
Ensemble = bagging(t,S_train,label_train,num_sample,varDict,attributes)
index = list(Ensemble)
for i in range(len(index)):
    cur_Ensemble = {}
    curInd = index[:i+1]
    for j in range(len(curInd)):
        cur_Ensemble[curInd[j]] = Ensemble[j]
        
    pred_train = predict_bagging(S_train,label_train,cur_Ensemble,varDict)
    pred_test = predict_bagging(S_test,label_test,cur_Ensemble,varDict)
    err_train.append(len(label_train[pred_train!=label_train])/len(label_train))
    err_test.append(len(label_test[pred_test!=label_test])/len(label_test))
    logging.basicConfig(filename='bagging.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG) 
    logging.info('{} depth - train err: {}, test err: {}'.format(i+1,err_train[-1],err_test[-1]))

print("Q2-2-b: END bagging")
plt.plot(err_train,label='train')
plt.plot(err_test,label='test')
plt.grid()
plt.legend()
plt.savefig('bagging.jpg')
