import numpy as np

def sampling_weight(data,label,weights,original_n):
    n = data.shape[0]
    
    choices = np.random.choice(n,original_n,p = weights)
    
    return data[choices],label[choices]
