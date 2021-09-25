import numpy as np
from IG3 import IG3
from entropy import entropy
from majority_error import majority_error
from gini import gini

S = np.array([['S','H','H','W'],
             ['S','H','H','S'],
             ['O','H','H','W'],
             ['R','M','H','W'],
             ['R','C','N','W'],
             ['R','C','N','S'],
             ['O','C','N','S'],
             ['S','M','H','W'],
             ['S','C','N','W'],
             ['R','M','N','W'],
             ['S','M','N','S'],
             ['O','M','H','S'],
             ['O','H','N','W'],
             ['R','M','H','S']])

label = np.array(['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No'])
v_dict = {'Outlook':['S','O','R'],'Temperature':['H','M','C'],'Humidity':['H','N','L'],'Wind':['S','W']}

print('**Majority Error**')
Tree = {}
parent = None
link = None
cur_depth = 1
maxDepth = None
majority = None
fn = majority_error
Tree2 = IG3(S,v_dict,label,cur_depth,maxDepth,fn,Tree,parent,link,majority)
print(Tree2)
print('**GINI**')
Tree = {}
parent = None
link = None
cur_depth = 1
maxDepth = None
majority = None
fn = gini
Tree3 = IG3(S,v_dict,label,cur_depth,maxDepth,fn,Tree,parent,link,majority)
print(Tree3)
print('**SAME TREE**')
print('ME and GINI: {}'.format(Tree2==Tree3))
