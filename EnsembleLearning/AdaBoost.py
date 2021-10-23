import numpy as np
from IG3 import *
from entropy import entropy
from gini import gini
from vote import vote
from update_weights import update_weights
from sampling_weight import sampling_weight

def AdaBoost(t,Data,label,varDict,attributes):
    Data = np.array(Data,copy=True)
    label = np.array(label,copy=True)
    n = Data.shape[0]
    weights = (1/n)*np.ones(n)
    
    Ensemble = {}
    for i in range(t):
        
        # Find Tree
        maxDepth = 1
        cur_depth = 1
        parent,edge = None,None
        curTree = Tree(maxDepth)
        curTree.is_weights = True
        fn = entropy
        
        IG3(curTree,Data,label,attributes,varDict,cur_depth,fn,parent,edge,weights)
        # Calculate vote
        cur_predict = prediction(curTree,Data,varDict)

        et = sum(weights[label!=cur_predict])
        at = vote(et)
        
        # Update weights
        weights = update_weights(at,weights,cur_predict,label)
        
        Ensemble[i] = {'Tree':curTree,'Vote':at}
        
    return Ensemble

def predict_Ensemble(Data,label,Ensemble,varDict):
    eles = np.unique(label)
    voting = np.zeros((len(label),len(eles)))
    
    for ind in Ensemble:
        curTree,curVote = Ensemble[ind]['Tree'],Ensemble[ind]['Vote']
        curPred = prediction(curTree,Data,varDict)
        for cur_ind,pred in enumerate(curPred):
            cur_j = eles.tolist().index(pred)
            voting[cur_ind,cur_j] += curVote

    election = eles[np.argmax(voting, axis=1)]
    
    return election
        
