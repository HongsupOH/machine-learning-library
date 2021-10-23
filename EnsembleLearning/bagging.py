import numpy as np
from IG3 import *
from entropy import entropy
from gini import gini

def bagging(t,Data,label,num_sample,varDict,attributes,att_subsets = None):
    Data = np.array(Data,copy=True)
    label = np.array(label,copy=True)
    n,m = Data.shape
    
    Ensemble = {}
    for i in range(t):
        
        # Sampling
        indexes = np.random.choice(n,num_sample)
        sample_data = Data[indexes]
        sample_label = label[indexes]
        # Generate Tree
        maxDepth = -1
        cur_depth = 1
        parent,edge = None,None
        curTree = Tree(maxDepth)
        fn = entropy
        
        IG3(curTree,sample_data,sample_label,attributes,varDict,\
            cur_depth,fn,parent,edge,num_att_subsets=att_subsets)
        
        #Save current tree 
        Ensemble[i] = {'Tree':curTree}

    return Ensemble

def predict_bagging(Data,label,Ensemble,varDict):
    eles = np.unique(label)
    voting = np.zeros((len(label),len(eles)))
    
    for ind in Ensemble:
        curTree = Ensemble[ind]['Tree']
        curPred = prediction(curTree,Data,varDict)
        for cur_ind,pred in enumerate(curPred):
            cur_j = eles.tolist().index(pred)
            voting[cur_ind,cur_j] += 1

    election = eles[np.argmax(voting, axis=1)]
    
    return election


    
