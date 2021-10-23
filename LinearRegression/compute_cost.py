import numpy as np

def compute_cost(X,Y,w):
    return 0.5*np.sum((Y - X.dot(w))**2)

