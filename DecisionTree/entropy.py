import numpy as np

def entropy(label,weights):
    n = label.shape[0]
    answer = 0
    elements = np.unique(label)
    ps = np.zeros(len(elements))
    for ind,element in enumerate(elements):
        
        ps[ind] = sum(weights[np.where(label==element)[0]])
        
    ps = ps/sum(ps)
    answer -= sum(ps*np.log2(ps))
    return answer

