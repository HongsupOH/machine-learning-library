import numpy as np

def modify_label(Y):
    for ind,y in enumerate(Y):
        if y == 0:
            Y[ind] = -1
    return Y

def gen_train_test(file):
    Data = np.loadtxt(file,delimiter=',',dtype=np.double)
    X,Y = Data[:,:-1],Data[:,-1]
    
    Data = np.loadtxt(file,delimiter=',',dtype=np.double)
    X,Y = Data[:,:-1],Data[:,-1]
    one = np.ones((len(X),1))
    X = np.hstack((one,X))
    
    return X,Y
