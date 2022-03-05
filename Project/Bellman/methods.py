"""
A collection of additional operations
"""

import numpy as np
import math as mt

def model_step(sys, action, x_prev):
    """
    Calculate the next state of the dynamical system $X_{n+1} = F(X_n, a_n)$
    
    Parameters
    ----------
    
    sys: object of the class Model (state matrices $A_1$ and $A_2$ are required)
    action: the decision $a_n$ (0: the sensor transmits, 0: the sensor does not transmit)

    Returns
    -------
    X: the next state $X_{n+1}$
    """
    X = (sys.A1 + sys.A2 * action) @ x_prev + 0.1 * np.array([[np.random.randn()], [0]])
    return X

def match_value(x,interval):
    """
    Returns the value from an interval closest to x.
    Operate as a quantizer
    
    Parameters
    ----------
    
    x: input value
    interval: ascending ordered set

    Returns
    -------
    value: the value from the interval closest to x
    index: the index of 'value' in 'interval'
    """
    if x <= interval[0]: #if smaller than the lower bound
        value = interval[0] #return the lower bound
        index = 0
        return value, index
    elif x >= interval[-1]: #if larger than the upper bound
        value = interval[-1] #return the upper bound
        index = -1
        return value, index
    
    for i in range(1,len(interval)): #find the closest value in the interval
        if x <= interval[i]:
            if (x-interval[i-1]) <= (interval[i]-x):
                value = interval[i-1]
                index = i-1
            else:
                value = interval[i]
                index = i
            return value, index
            
def generate_sample(set_of_values, set_of_probabilities):
    """
    Generate a sample from given distribution
    
    Parameters
    ----------
    
    set_of_values: vector of values
    set_of_probabilities: corresponding vector of probabilities

    Returns
    -------
    an element from 'set_of_values'
    """
    y = np.random.rand()
    val = 0;
    for j in range(0,len(set_of_probabilities)):
        val = val + set_of_probabilities[j]
        if y <= val:
            i = j
            break
    return set_of_values[i]     
        
def mapping(x0, X_ss, T):
    """
    One step of mapping of sampled system which is a stochastic
    approximation of the dynamical system
    
    Parameters
    ----------
    
    x0: previous value (vector, numpy ndarray)
    X_ss: discrete-state space (numpy ndarray)
    T: transition probability matrix (numpy ndarray)

    Returns
    -------
    next value $X_{n+1}$
    """
    # Generate next sample of the model
    n = len(X_ss)
    _, index1 = match_value(x0[0],X_ss)
    _, index2 = match_value(x0[1],X_ss)
    
    nonzero_probability_indices = np.flatnonzero(T[index1, index2,:,:])
    nonzero_probability_values = np.ndarray.flatten(T[index1, index2,:,:])[nonzero_probability_indices]
    value = generate_sample(nonzero_probability_indices,nonzero_probability_values)
    i, j = np.unravel_index(value,(n,n))
    return np.array([[X_ss[i]],[X_ss[j]]])