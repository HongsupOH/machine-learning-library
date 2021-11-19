import numpy as np
import scipy.optimize
import scipy.spatial.distance

def objectiveFN(X,Y,alpha,args):
    kernelFN,g = args[0],args[1]
    kij = kernelFN(X,X,g=g)
    aayykij = np.outer(alpha,alpha)*np.outer(Y,Y)*kij
    J1 = np.sum(aayykij)
    J2 = sum(alpha)
    J = (1/2)*J1 - J2
    return J

def optimization(fn,kernel,alpha,g,C,Y):
    # Define boundary
    bound = _bounds(C,Y)
    bound = tuple(bound)
    # Define constraints
    
    constraint = {'type':'eq','fun':_constraints,'args':(Y,)}
    print('Start optimize...')
    
    result = scipy.optimize.minimize(fn,alpha,\
                                     args=[kernel,g],
                                     method='SLSQP',\
                                     bounds = bound,\
                                     constraints = constraint)
    alpha = np.array(result.x)
    return alpha

def _bounds(C,Y):
    return [(0,C)]*len(Y)

def _constraints(alpha,Y):
    return alpha.dot(Y)

def Kernel(X,Z,g=0):
    if g == 0:
        G = X.dot(Z.T)
        return G
    else:
        sq_dist = scipy.spatial.distance.cdist(X,Z,'sqeuclidean')
        G = np.exp(-(sq_dist/g))
        return G

def dual_weights(alpha,X,Y):
    return np.sum((alpha*Y).reshape(-1,1)*X,axis=0)
    
    
def dual_bias(alpha,X,Y,g):
    ind = np.where(alpha>0)[0]
    b = np.array(Y[ind] - np.sum((alpha*Y).reshape(-1,1)*Kernel(X,X[ind],g=g),axis=0))
    return np.mean(b)
    
def dual_prediction(w,b,X): 
    return np.sign(b + X.dot(w))

def dual_prediction_kernel(alpha,X,Z,Y,b,g): 
    return np.sign(np.sum((alpha*Y).reshape(-1,1)*Kernel(X,Z,g=g),axis=0))
    
    










    

    


