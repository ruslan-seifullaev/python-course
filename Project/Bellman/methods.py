import numpy as np
import math as mt
import math as mt

def model_step(sys, action, x_prev):
    X = (sys.A1 + sys.A2 * action) @ x_prev + 0.1 * np.array([[np.random.randn()], [0]])
    return X

def match_value(x,interval):
    if x <= interval[0]:
        value = interval[0]
        index = 0
        return value, index
    elif x >= interval[-1]:
        value = interval[-1]
        index = -1
        return value, index
    
    for i in range(1,len(interval)):
        if x <= interval[i]:
            if (x-interval[i-1]) <= (interval[i]-x):
                value = interval[i-1]
                index = i-1
            else:
                value = interval[i]
                index = i
            return value, index
            
def generate_sample(set_of_values, set_of_probabilities):
    #set_of_values: vector of values
    #set_of_probabilities:   vector of probabilities
    y = np.random.rand()
    val = 0;
    for j in range(0,len(set_of_probabilities)):
        val = val + set_of_probabilities[j]
        if y <= val:
            i = j
            break
    return set_of_values[i]     
        
def mapping(x0, X_ss, T):
    # Generate next sample of the model
    n = len(X_ss)
    _, index1 = match_value(x0[0],X_ss)
    _, index2 = match_value(x0[1],X_ss)
    
    nonzero_probability_indices = np.flatnonzero(T[index1, index2,:,:])
    nonzero_probability_values = np.ndarray.flatten(T[index1, index2,:,:])[nonzero_probability_indices]
    value = generate_sample(nonzero_probability_indices,nonzero_probability_values)
    i, j = np.unravel_index(value,(n,n))
    return np.array([[X_ss[i]],[X_ss[j]]])