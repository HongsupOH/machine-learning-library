import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging

sys.path.append('../../DecisionTree')
sys.path.append('../../EnsembleLearning')

from helper import *
from AdaBoost import *

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
print("Q2-2-a: START Adaboost")
err_train_part = []
err_test_part = []

t = 500

Ensemble = AdaBoost(t,S_train,label_train,varDict,attributes)
index = list(Ensemble)

for i in range(len(index)):
    cur_Ensemble = {}
    curInd = index[:i+1]
    for j in range(len(curInd)):
        cur_Ensemble[curInd[j]] = Ensemble[j]
    pred_train = predict_Ensemble(S_train,label_train,cur_Ensemble,varDict)
    pred_test = predict_Ensemble(S_test,label_test,cur_Ensemble,varDict)

    tmp_train_part = []
    tmp_test_part = []
    for key in cur_Ensemble:
        tree = Ensemble[key]['Tree']
        y_train = prediction(tree,S_train,varDict)
        tmp_train_part.append(len(label_train[label_train!=y_train])/len(label_train))
        y_test = prediction(tree,S_test,varDict)
        tmp_test_part.append(len(label_test[label_test!=y_test])/len(label_test))
        
    err_train_part.append(tmp_train_part)
    err_test_part.append(tmp_test_part)

    err_train.append(len(label_train[pred_train!=label_train])/len(label_train))
    err_test.append(len(label_test[pred_test!=label_test])/len(label_test))
    logging.basicConfig(filename='Adaboost.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG) 
    logging.info('{} depth - train err: {}, test err: {}'.format(index[i],err_train[-1],err_test[-1]))

print("Q2-2-a: END Adaboost")

plt.figure()
plt.plot(err_train,label='Train')
plt.plot(err_test,label='Test')
plt.grid()
plt.legend()
plt.savefig('AdaBoost.jpg')

plt.figure(figsize=(100,10))

for i in range(len(err_train_part)):
    plt.boxplot(err_train_part[i],positions = [(i+1)-0.2])
    plt.boxplot(err_test_part[i],positions = [(i+1)+0.2])

plt.xticks([x+1 for x in range(len(err_train_part))])
plt.grid()
plt.legend(['train','test'])
plt.savefig('AdaBoost_stump_err.jpg')





