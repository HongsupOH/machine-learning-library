import numpy as np

def gini(label):
    n = label.shape[0]
    unique,counts = np.unique(label,return_counts = True)
    ps = counts/n
    ps_sq = ps**2
    answer = 1-sum(ps_sq)
    return answer
