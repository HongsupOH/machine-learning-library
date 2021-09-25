import numpy as np

def entropy(label):
    n = label.shape[0]
    answer = 0
    unique,counts = np.unique(label,return_counts = True)
    ps = counts/n
    answer -= sum(ps*np.log2(ps))
    return answer

