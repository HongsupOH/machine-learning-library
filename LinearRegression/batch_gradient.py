import numpy as np
from compute_cost import *

def batch_gradient(X,Y,T,r,threshold):
    X,Y = np.array(X,copy=True),np.array(Y,copy=True)
    n,m = X.shape
    while True:
        
        w = np.zeros((T+1,m))
        costs = np.zeros(T+1)
        costs[0] = compute_cost(X,Y,w[0])
        for t in range(T):
            gradient = compute_gradient(X,Y,w[t])
            w[t+1] = w[t] - r*gradient
            costs[t+1] = compute_cost(X,Y,w[t+1])
            if np.linalg.norm(w[t+1]-w[t])<=threshold:
                print('Convergence at r = {}, T = {}'.format(r,t))
                return w[t+1],costs

        print("Fail to converge at {}".format(r))
        r*= 0.5
        
    return 

def compute_gradient(X,Y,w):
    h = X.dot(w)
    error = Y - h
    return -X.T.dot(error)


    
    
