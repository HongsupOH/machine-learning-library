import numpy as np
from entropy import *

def information_gain(fn,A,label,v_s,weights):
    n = label.shape[0]
    answer = fn(label, weights)
    for v in v_s:
        label_v = label[A == v]
        weights_v = weights[A == v]
        m = label_v.shape[0]
        if m != 0:
            p = m/n
            answer -= p*fn(label_v, weights_v)
    return answer


