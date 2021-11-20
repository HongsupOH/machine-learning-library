import numpy as np

def stochastic_sub_gradient_descent(X,Y,C,r0,a,epoch,mode):
    n,m = X.shape
    w = np.zeros(m)
    for t in range(epoch):
        rt = update_gamma(r0,a,t,mode)
        data = np.hstack((X,Y.reshape(len(Y),1)))
        np.random.shuffle(data)
        X_s,Y_s = data[:,:-1],data[:,-1]
        for ind,x in enumerate(X_s):
            y = Y_s[ind]
            pred = y*(w.dot(x))
            if pred <=1:
                w0 = np.array(w,copy=True)
                w0[-1] = 0
                w = w - rt*w0 +rt*C*n*y*x
            else:
                w[:-1] = (1 - rt)*w[:-1]
    return w

def update_gamma(r0,a,t,mode):
    if mode == 1:
        return r0/(1+(r0/a)*t)
    elif mode == 2:
        return r0/(1+t)


def prediction(w,X):
    return np.sign(X.dot(w))
