import numpy as np
from collections import defaultdict
from IG3 import *
from entropy import entropy
from information_gain import information_gain

S = np.array([[0,0,1,0],
              [0,1,0,0],
              [0,0,1,1],
              [1,0,0,1],
              [0,1,1,0],
              [1,1,0,0],
              [0,1,0,1]])



label = np.array([0,0,1,1,0,0,0])
v_dict = {'x1':[0,1],'x2':[0,1],'x3':[0,1],'x4':[0,1]}
fn = entropy

visit = set()
Tree = {}
parent = None
link = None
cur_depth = 1
maxDepth = None
majority = None

Tree = IG3(S,v_dict,label,cur_depth,maxDepth,fn,Tree,parent,link,majority)
print(Tree)
y_pred = predict(Tree,S,label,v_dict)
print(y_pred)
print(error(label,y_pred))
