import numpy as np

def perceptron_average(X,Y,r,epoch):
    n,m = X.shape
    Y = modify_label(Y)
    w = np.zeros(m)
    a = np.zeros(m)
    for t in range(epoch):
        for ind,x in enumerate(X):
            y = Y[ind]
            pred = y*(w.dot(x))
            if pred<=0:
                w += r*y*x
            a += w
            
    return a


def prediction_average(a,X):
    
    return np.sign(X.dot(a))


def modify_label(Y):
    for ind,y in enumerate(Y):
        if y == 0:
            Y[ind] = -1
    return Y
