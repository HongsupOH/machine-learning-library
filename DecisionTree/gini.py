import numpy as np

def gini(label,weights):
    n = label.shape[0]
    
    elements = np.unique(label)
    ps = np.zeros(len(elements))
    
    for ind,element in enumerate(elements):
        ps[ind] += sum(weights[np.where(label==element)[0]])
    
    ps = ps/sum(ps)
    ps_sq = ps**2
    answer = 1-sum(ps_sq)
    return answer
