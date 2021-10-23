import numpy as np
from IG3 import *

def open_file(file):
    f = open(file)
    data = []
    for line in f:
        term = line.strip().split(',')
        data.append(term)
    data = np.array(data)

    S = data[:,:-1]
    label = data[:,-1]
    return S,label

def numericalAttribute(S_train,S_test,A,ind):
    #Train data
    A_tmp = np.array(A,copy=True,dtype=float)
    med = np.median(A_tmp)
    A[A_tmp<med] = 'less:{}'.format(med)
    A[A_tmp>=med] = 'more:{}'.format(med)
    S_train[:,ind] = A
    Att = ['less:{}'.format(med),'more:{}'.format(med)]
    #Test data
    A_tmp2 = np.array(S_test[:,ind],copy=True,dtype=float)
    A_test = np.array(S_test[:,ind],copy=True)
    A_test[A_tmp2<med] = 'less:{}'.format(med)
    A_test[A_tmp2>=med] = 'more:{}'.format(med)
    S_test[:,ind] = A_test
    return Att


def missingData(S):
    S = np.array(S,copy=True)
    S = S.T
    for ind,row in enumerate(S):
        if 'unknown' not in row.tolist():
            pass
        else:
            unique,counts = np.unique(row,return_counts = True)
            unk_ind = unique.tolist().index('unknown')
            unique = np.delete(unique,unk_ind)
            counts = np.delete(counts,unk_ind)
            majority = unique[np.argmax(counts)]
            row[row == 'unknown'] = majority
            S[ind] = row
    S = S.T
    return S




            
