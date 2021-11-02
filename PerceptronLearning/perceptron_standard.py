import numpy as np

def perceptron_standard(X,Y,r,epoch):
    n,m = X.shape
    Y = modify_label(Y)
    w = np.zeros(m)
    for t in range(epoch):
        X_s,Y_s = shuffle(X,Y)
        for ind,x in enumerate(X_s):
            y = Y_s[ind]
            pred = y*(w.dot(x))
            if pred<=0:
                w += r*y*x
    return w

def prediction_standard(w,X):
    return np.sign(X.dot(w))

def modify_label(Y):
    for ind,y in enumerate(Y):
        if y == 0:
            Y[ind] = -1
    return Y


def shuffle(X,Y):
    X,Y = np.array(X,copy=True),np.array(Y,copy=True)
    data = np.hstack((X,Y.reshape((len(Y),1))))
    np.random.shuffle(data)
    X_s,Y_s = data[:,:-1],data[:,-1]
    return X_s,Y_s
