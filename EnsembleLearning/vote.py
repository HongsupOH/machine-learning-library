import numpy as np

def vote(et):
    
    at = (1/2)*np.log2((1-et)/et)
    
    return at
