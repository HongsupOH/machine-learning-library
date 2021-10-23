import numpy as np

def update_weights(at,weights,predict,target):
    
    for ind,w in enumerate(weights):
        p,t = predict[ind],target[ind]
        if p!=t:
            weights[ind] = w*np.exp(at)
        else:
            weights[ind] = w*np.exp(-at)
    
    weights = weights/sum(weights)
    
    return weights
