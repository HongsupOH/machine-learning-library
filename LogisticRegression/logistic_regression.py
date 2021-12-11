import numpy as np

def logistic_regression_MAP(X,Y,epoch,r0,a,v,mode):
    n,m = X.shape
    w = np.zeros(m)
    Losses = []
    for t in range(epoch):
        rt = update_gamma(r0,a,t,mode)
        data = np.hstack((X,Y.reshape(len(Y),1)))
        np.random.shuffle(data)
        Xs,Ys = data[:,:-1],data[:,-1]
        y_hats = []
        dw = 0
        for ind,x in enumerate(Xs):
            y = Ys[ind]
            z = y*(w.dot(x))
            s = sigmoid(z)
            dw = -s*(1-s)*y*x*m + (1/(v**2))*w
            w -= rt*dw
            
    return w

def logistic_regression_ML(X,Y,epoch,r0,a,mode):
    n,m = X.shape
    w = np.zeros(m)
    Losses = []
    for t in range(epoch):
        rt = update_gamma(r0,a,t,mode)
        data = np.hstack((X,Y.reshape(len(Y),1)))
        np.random.shuffle(data)
        Xs,Ys = data[:,:-1],data[:,-1]
        y_hats = []
        dw = 0
        for ind,x in enumerate(Xs):
            y = Ys[ind]
            z = y*(w.dot(x))
            s = sigmoid(z)
            dw = -s*(1-s)*y*x*m 
            w -= rt*dw
            
    return w
            
def sigmoid(z):
    return 1/(1+np.exp(-z))

def update_gamma(r0,a,t,mode):
    if mode == 1:
        return r0/(1+(r0/a)*t)
    elif mode == 2:
        return r0/(1+t)


def prediction(w,X):
    y = sigmoid(X.dot(w))
    y[y>=0.5] = 1
    y[y<0.5] = -1
    return y
