import numpy as np

def majority_error(label):
    n = label.shape[0]
    unique,counts = np.unique(label,return_counts = True)
    maxCounts = max(counts)
    p = maxCounts/n
    majorityError = 1 - p
    return majorityError
