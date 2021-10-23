import numpy as np

def majority_error(label,weights):
    n = label.shape[0]
    elements = np.unique(label)
    ps = np.zeros(len(elements))
    for ind,element in enumerate(elements):
        ps[ind] += sum(weights[np.where(label==element)[0]])
    ps = ps/sum(ps)
    p = max(ps)
    majorityError = 1 - p
    return majorityError
