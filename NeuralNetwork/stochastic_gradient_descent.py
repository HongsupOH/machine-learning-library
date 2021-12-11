import numpy as np


def stochastic_gradient_descent(X,Y,epoch,NN,r0,a,mode):
    n,m = X.shape
    Losses = []
    for t in range(epoch):
        rt = update_gamma(r0,a,t,mode)
        data = np.hstack((X,Y.reshape(len(Y),1)))
        np.random.shuffle(data)
        Xs,Ys = data[:,:-1],data[:,-1]
        y_hats = []
        for ind,x in enumerate(Xs):
            x = np.array([x])
            y = np.array([Ys[ind]])
            
            NN.X = x
            
            y_hat = NN.forward()
            y_hats.append(y_hat[0,0])
            
            NN.backward(y,y_hat)
            
            #Update w
            for layer in NN.structure:
                w = NN.structure[layer].weight
                dw = NN.structure[layer].der_weight
               
                w += rt*dw
                
                NN.structure[layer].weight = w
                NN.structure[layer].der_weight = None
                
        Loss = (1/2)*sum((Y - y_hats)**2)
        
        Losses.append(Loss)
        if len(Losses)>2:
            diff = abs(Losses[-1] - Losses[-2])
            
            if diff <= 1e-5:
                return Losses
                
    return Losses
    
            

def update_gamma(r0,a,t,mode):
    if mode == 1:
        return r0/(1+(r0/a)*t)
    elif mode == 2:
        return r0/(1+t)
