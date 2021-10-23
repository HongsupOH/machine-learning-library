import numpy as np
import logging
from IG3 import *
from bagging import *
from entropy import entropy
from gini import gini

def bias_variance_decomposition(Data,Data_test,label,varDict,attributes,iteration,num_sample,t,att_subsets = None):

    label_numeric = _label_modification(label)
    
    single_labels = np.zeros((len(label),iteration))
    ensemble_labels = np.zeros((len(label),iteration))

    for ind in range(iteration):
        
        Ensemble = bagging(t,Data,label,num_sample,varDict,attributes,att_subsets=att_subsets)
        single_label = _label_modification(prediction(Ensemble[0]['Tree'],Data_test,varDict))
        ensemble_label = _label_modification(predict_bagging(Data_test,label,Ensemble,varDict))

        single_labels[:,ind] = single_label
        ensemble_labels[:,ind] = ensemble_label

        logging.basicConfig(filename='bias_variance_subset={}.log'.format(att_subsets), filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        logger=logging.getLogger()
        logger.setLevel(logging.DEBUG) 
        logging.info('Finish {}/{}'.format(ind+1,iteration))

    single_biases, ensemble_biases = [],[] 
    single_variances, ensemble_variances = [],[]
    
    for ind,single_row in enumerate(single_labels):
        ensemble_row = ensemble_labels[ind]
        
        single_avg = np.average(single_label)
        ensemble_avg = np.average(ensemble_label)
        
        single_biases.append((single_avg - label_numeric[ind])**2)
        ensemble_biases.append((ensemble_avg - label_numeric[ind])**2)
        
        single_variances.append(np.var(single_row))
        ensemble_variances.append(np.var(ensemble_row))

    single_bias = np.average(single_biases)
    single_variance = np.average(single_variances)
    single_square = single_bias + single_variance

    ensemble_bias = np.average(ensemble_biases)
    ensemble_variance = np.average(ensemble_variances)
    ensemble_square = ensemble_bias + ensemble_variance

    return single_bias, single_variance, single_square, ensemble_bias, ensemble_variance, ensemble_square
    
def _label_modification(label):
    
    numeric_label = np.zeros(len(label))
    
    for i,y in enumerate(label):
        
        if y == 'yes':
            numeric_label[i] = 1
        else:
            numeric_label[i] = -1
            
    return numeric_label
        
