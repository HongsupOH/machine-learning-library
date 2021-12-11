import numpy as np

class NeuralNetwork(object):

    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.level = 0
        self.depth = 0
        self.structure = {}

    def update_weight(self,weight,level):
        """
        weight: given weight
        """
        self.structure[level].weight = weight
        

    def zero_weight(self,n,width):
        """
        **Generate Initial weight**
        n: prev-node shape
        width: next-node shape
        """
        return np.zeros((n,width))

    def uniform_weight(self,n,width):
        """
        **Generate Initial weight**
        n: prev-node shape
        width: next-node shape
        """
        return np.random.rand((n,width))

    def gaussian_standard_weight(self,n,width):
        """
        **Generate Initial weight**
        n: prev-node shape
        width: next-node shape
        """
        return np.random.normal(0,1, size = (n,width))

    def linear(self,weight,z):
        """
        **Linear Layer**
        width: next-node shape
        z: input data
        """
        a = z.dot(weight)
        return a
    
    def sigmoid(self,a):
        """
        a: linear data
        """
        z = 1/(1+np.exp(-a))
        return z
        

    def add_layer(self,width,activation,distribution):
        """
        width: width of current layer
        activation: sigmoid, RELU,...,etc
        distribution: initial weight distribution
        """
        nn_layer = Layer(width)
        if self.level == 0:
            n = self.X.shape[1]
            nn_layer.weight = distribution(n,width)
        else:
            n = self.structure[self.level-1].weight.shape[1] + 1
            nn_layer.weight = distribution(n,width)

        nn_layer.linear = self.linear
        nn_layer.activation = activation
        
        self.structure[self.level] = nn_layer
        self.level += 1
        self.depth += 1

    def forward(self):
        
        for level,layer in self.structure.items():
            
            if level==0:
                z = self.X
                
            weight = layer.weight
            
            a = layer.linear(weight,z)
            self.structure[level].a = a
            
            if self.structure[level].activation != self.linear:
                z = layer.activation(a)
                ones = np.ones((z.shape[0],1))
                z = np.hstack((ones,z))
                self.structure[level].z = z
            else:
                self.structure[level].z = a
                
        return a

    def prediction(self,X):
        predict = []
        for x in X:
            self.X = np.array([x])
            y_hat = self.forward()
            predict.append(np.sign(y_hat[0,0]))
        return predict
            

    def backward(self,label,predict):
        stack = [label - predict]
        layers = list(self.structure)[::-1]
        for ind,layer in enumerate(layers):
        
            prev_der = stack.pop()
            
            dLdW = self.der_by_w(layer, prev_der)
            self.structure[layer].der_weight = dLdW
            dLdz = self.der_by_z(layer, prev_der)
            stack.append(dLdz)
            
            
    def der_by_w(self,level,prev_der):
        if level==0:
            z_prev = self.X
        else:
            z_prev = self.structure[level-1].z
            
        z = self.structure[level].z
        w = self.structure[level].weight
        
        n,m = w.shape[0],w.shape[1]
        dLdW = np.zeros((n,m))

        row_ele = np.arange(0,n)
        ind_row = np.array(np.kron(row_ele,np.ones(m)),dtype=int)
        col_ele = np.arange(0,m)
        ind_col = np.array(col_ele.tolist()*n,dtype=int)
        
        if self.structure[level].activation == self.sigmoid:
            dLdW[ind_row,ind_col] = z[:,ind_col+1]*(1-z[:,ind_col+1])*z_prev[:,ind_row]
        else:
            dLdW[ind_row,ind_col] = z_prev[:,ind_row]

        ind_der = np.arange(0,len(prev_der))
        dLdW[:,ind_der] *= prev_der[ind_der,:].reshape(-1)
        
        return dLdW

    def der_by_z(self,level,prev_der):
        if level==0:
            z_prev = self.X
        else:
            z_prev = self.structure[level-1].z
            
        z = self.structure[level].z
        if level != self.depth-1:
            z_prev = z_prev[:,1:]
            z = z[:,1:]
        
        w = self.structure[level].weight
        
        n,m = z_prev.shape[1],len(prev_der)
        
        dz = np.zeros((n,m))

        row_ele = np.arange(0,n)
        ind_row = np.array(np.kron(row_ele,np.ones(m)),dtype=int)
        col_ele = np.arange(0,m)
        ind_col = np.array(col_ele.tolist()*n,dtype=int)
        
        if self.structure[level].activation == self.sigmoid:
            dz[ind_row, ind_col] = z[:,ind_col]*(1 - z[:,ind_col])*w[ind_row+1,ind_col]

           
        else:
            dz[ind_row, ind_col] = w[ind_row,ind_col]
        

        dLdz = dz.dot(prev_der)
        if level == self.depth-1:
            dLdz = dLdz[1:]
        
        return dLdz
    

class Layer(object):

    def __init__(self,width):
        self.width = width
        self.weight = None
        self.der_weight = None
        self.linear = None
        self.activation = None
        self.a = None
        self.z = None



















        
