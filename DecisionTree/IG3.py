import numpy as np
import random
from information_gain import *

def IG3(S,v_dict,label,cur_depth,maxDepth,fn,Tree,parent,link,majority):
    
    # Maximum Depth
    if maxDepth!=None:
        if cur_depth > maxDepth:
            return Tree
    
    keys = list(v_dict)
    
    unique,counts = np.unique(label,return_counts = True)
    if unique.shape[0] == 0:
        if Tree[parent]['child']==None:
            Tree[parent]['child'] = {link: {'label':majority}}
        else:
            Tree[parent]['child'][link] = {'label':majority}
    elif unique.shape[0] == 1:
        
        if Tree[parent]['child']==None:
            Tree[parent]['child'] = {link: {'label':unique[0]}}
        else:
            Tree[parent]['child'][link] = {'label':unique[0]}
    else:
        #Find maximum index of information gain
        maxIG,maxInd = -np.inf,None
        IGs = []
        for ind in range(S.shape[1]):
            A = S[:,ind]
            v_s = v_dict[keys[ind]]
            
            curIG = information_gain(fn,A,label,v_s)
            
            IGs.append(curIG)
            
        #maxInd = IGs.index(max(IGs))
        if max(IGs)<=1e-10:
            maxInd = random.randint(0,len(IGs)-1)
        else:
            if IGs.count(max(IGs))>1:
                IGs = np.array(IGs)
                ids = np.where(IGs==max(IGs))[0]
                maxInd = random.choice(ids)
            else:
                maxInd = IGs.index(max(IGs))
        
        #Add visited node    
        if parent == None:
            curNode = ((link,keys[maxInd]),)
            Tree[curNode] = {'depth':cur_depth,'parent':parent,'child':None}
            Tree['root'] = ((link,keys[maxInd]),)
        else:
            plist = list(parent)
            plist += [(link,keys[maxInd])]
            curNode = tuple(plist)
            Tree[curNode] = {'depth':cur_depth,'parent':parent,'child':None}
            
            if Tree[parent]['child'] == None:
                Tree[parent]['child'] = {link:(link,keys[maxInd])}
            else:
                Tree[parent]['child'][link] = (link,keys[maxInd])

        #Find majority
        unique,counts = np.unique(label,return_counts = True)
        majority = unique[np.argmax(counts)]
        
        for edge in v_dict[keys[maxInd]]:
            S_v = S[S[:,maxInd]==edge]
            label_v = label[S[:,maxInd]==edge]
            if len(label_v)==0:
                majority_v = majority
            else:
                unique,counts = np.unique(label_v,return_counts = True)
                majority_v = unique[np.argmax(counts)]
            
            if cur_depth == maxDepth:
                
                if Tree[curNode]['child']==None:
                    Tree[curNode]['child'] ={edge:{'label':majority_v}}
                else:
                    Tree[curNode]['child'][edge] = {'label':majority_v}
            
            IG3(S_v,v_dict,label_v,cur_depth+1,maxDepth,fn,Tree,curNode,edge,majority)

    return Tree

def predict(Tree,S,label,v_dict):
    y_pred = []
    root = Tree['root']
    for ind,row in enumerate(S):
        
        predict_help(Tree,root,row,v_dict,y_pred)
        
    y_pred = np.array(y_pred)
    
    return y_pred

def predict_help(Tree,root,row,v_dict,y_pred):
    
    keys = list(v_dict)
    ind = keys.index(root[-1][1])
    v = row[ind]
    
    if v in Tree[root]['child']:
        child_att = Tree[root]['child'][v]
        if 'label' in child_att:
            y = child_att['label']
            y_pred.append(y)
            
            return
        else:
            
            node = list(root) + [child_att]
            predict_help(Tree,tuple(node),row,v_dict,y_pred)
        

def error(label,predict):
    diff = predict==label
    return 1-sum(diff)/len(diff)
