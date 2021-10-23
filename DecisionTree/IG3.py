import numpy as np
import random
import heapq
from Tree import*
from information_gain import *
from Tree import*
from information_gain import *

def IG3(Tree,X,Y,attributes,varDict,cur_depth,fn,parent,edge,\
        weights=None, num_att_subsets=None):
   
    #1.No label (leaf)
    if len(Y)==0 or len(attributes)==0:
        curNode = Node('Leaf',None)
        curNode.is_leaf =  True
        curNode.label = parent.majority
        parent.get_child(curNode,edge)
        
    #2.Pure label (leaf)
    elif np.all(Y==Y[0]):
        curNode = Node('Leaf',None)
        curNode.is_leaf =  True
        curNode.label = Y[0]
        parent.get_child(curNode,edge)
        
    #3.Node
    else:
        variables = np.array(attributes,copy=True)
        weights = update_weights(Tree,Y,weights)
        
        if num_att_subsets!=None and X.shape[1] >= num_att_subsets:
            X_sub,inds = update_subsets(X,num_att_subsets)
            X_origin = np.array(X,copy=True)
            X = np.array(X_sub,copy=True)
            variables_origin = np.array(variables,copy=True)
            variables = variables[inds]
            
        #Find maximum index of information gain    
        IGs = []
        for ind in range(X.shape[1]):
            A = X[:,ind]
            categories = varDict[variables[ind]]
            curIG = information_gain(fn,A,Y,categories,weights)
            IGs.append(curIG)
        
        #Define current node
        curInd = np.argmax(IGs)
        node = variables[curInd]
        
        #Update Tree
        curNode = Node(node,cur_depth)
        if Tree.root == None:
            Tree.get_root(curNode)
        else:
            parent.get_child(curNode,edge)
            
        cur_attributes = varDict[node]
        
        majority = _majority(Y,weights/sum(weights))
        curNode.majority = majority
        
        if num_att_subsets!=None and X.shape[1] >= num_att_subsets:
            X = np.array(X_origin,copy=True)
            variables = np.array(variables_origin,copy=True)
            curInd = variables.tolist().index(node)
            
        #Check maximum depth
        if Tree.maxDepth == curNode.curDepth:
            for cur_attribute in cur_attributes:
                Y_sub = Y[X[:,curInd]==cur_attribute]
                weights_sub = weights[X[:,curInd]==cur_attribute]
                if len(Y_sub)==0:
                    pass
                else:
                    majority = _majority(Y_sub,weights_sub/sum(weights_sub))
                    
                subNode = Node('Leaf',None)
                subNode.is_leaf =  True
                subNode.label = majority
                curNode.get_child(subNode,cur_attribute)
        else:
            
            #Update child node
            for cur_attribute in cur_attributes:
                X_sub = X[X[:,curInd]==cur_attribute]
                X_sub = np.delete(X_sub,[curInd],axis=1)
                
                attributes_sub = np.delete(variables,[curInd])
                
                Y_sub = Y[X[:,curInd]==cur_attribute]
                weights_sub = weights[X[:,curInd]==cur_attribute]
                
                IG3(Tree,X_sub,Y_sub,attributes_sub,varDict,cur_depth+1,fn,curNode,cur_attribute,\
                    weights_sub/sum(weights_sub),num_att_subsets=num_att_subsets)

        
    return 

def prediction(tree,X,varDict):
    y_predict = []
    attributes = list(varDict)
    for ind,row in enumerate(X):
        y_predict.append(bfs(tree,attributes,row))
    y_predict = np.array(y_predict)
    return y_predict

def bfs(tree,attributes,row):
    queue = [tree.root]
    while queue:
        node = queue.pop(0)
        if node.is_leaf == True:
            return node.label
        else:
            ind = attributes.index(node.val)
            v = row[ind]
            queue.append(node.children[v])
    
def _majority(label,weights):
    n = label.shape[0]
    elements = np.unique(label)
    ps = np.zeros(len(elements))
    for ind,element in enumerate(elements):
        
        ps[ind] += sum(weights[np.where(label==element)[0]])

    return elements[np.argmax(ps)]

def update_weights(Tree,label,weights = None):
    n = len(label)
    if Tree.is_weights == False:
        weights = (1/n)*np.ones(n)
    else:
        weights = weights
    return weights

def update_subsets(X,num_att_subsets):
    n = X.shape[1]
    inds = np.random.choice(n,num_att_subsets,replace=False)
    X_sub = X[:,inds]
    return X_sub,inds

def error(label,predict):
    diff = predict==label
    return 1-sum(diff)/len(diff)

    
    
    
