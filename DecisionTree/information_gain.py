import numpy as np
from entropy import *

def information_gain(fn,A,label,v_s):
    n = label.shape[0]
    answer = fn(label)
    for v in v_s:
        label_v = label[A == v]
        m = label_v.shape[0]
        if m != 0:
            p = m/n
            answer -= p*fn(label_v)
    return answer


