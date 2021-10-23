import numpy as np
from compute_cost import *

def stochastic_gradient(X,Y,T,r,threshold):
    X,Y = np.array(X,copy=True),np.array(Y,copy=True)
    data = np.hstack((X,Y.reshape((len(Y),1))))
    
    n,m = X.shape
    while True:
        w = np.zeros((T+1,m))
        costs = np.zeros(T+1)
        costs[0] = compute_cost(X,Y,w[0])
        for t in range(T):
            np.random.shuffle(data)
            X_s,Y_s = data[:,:-1],data[:,-1]
    
            for i in range(n):
                error = Y_s[i] - X_s[i].dot(w[t])
                for j in range(m):
                    w[t+1,j] = w[t,j] + r*(error)*X_s[i,j]
                    
            costs[t+1] = compute_cost(X,Y,w[t+1])
            if np.abs(costs[t+1]-costs[t])<threshold:
                print("Convergence at {} iteration for r = {}".format(t+1,r))
                return w[t+1],costs

        print("Fail to converge at {}".format(r))
        r*=1/2

    return 
