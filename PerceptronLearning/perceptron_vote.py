import numpy as np

def perceptron_vote(X,Y,r,epoch):
    n,m = X.shape
    Y = modify_label(Y)
    W = np.zeros((n*epoch,m))
    C = np.zeros(n*epoch)
    k = 0
    for t in range(epoch):
        for ind,x in enumerate(X):
            y = Y[ind]
            pred = y*(W[k].dot(x))
            if pred<=0:
                W[k+1] = W[k] + r*y*x
                k += 1
                C[k] = 1
            else:
                C[k] += 1
    
    return W[:k+1],C[:k+1]


def prediction_vote(W,C,X):
    SUM = 0
    for ind,w in enumerate(W):
        SUM += C[ind]*np.sign(X.dot(w))
    return np.sign(SUM)


def modify_label(Y):
    for ind,y in enumerate(Y):
        if y == 0:
            Y[ind] = -1
    return Y

