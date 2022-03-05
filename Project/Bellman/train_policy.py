import numpy as np
import math as mt
from .methods import *

def value_iteration_algorithm(sys,discount_factor,balance_parameter,N_steps):
    """
    The value-iteration algorithm to solve the Bellman equation
    
    
    Parameters
    ----------
    
    sys: object of the class Model (the full state is: the dynamical system state X 
         the battery level, harvesting energy, transmission energy)
    discount_factor: the discount parameter in the algorithm
    balance_parameter: the balance parameter between the output penalty and the energy cost
    N_steps: number of steps in the algorithm

    Returns
    -------
    policy: numpy ndarray with (sub)optimal policy
    """
    n_X = len(sys.X_ss) #number of elements in the system state-space
    n_B = len(sys.B_ss) #number of elements in the state-space of the battery model
    n_H = len(sys.H_ss) #number of elements in the state-space of the harvesting model
    n_T = len(sys.T_ss) #number of elements in the state-space of the transmission model
    
    V = np.zeros((n_X,n_X,n_B,n_H,n_T)) #initialize the value function
    policy = np.zeros((n_X,n_X,n_B,n_H,n_T)) #initialize the policy
    
    for N in range(N_steps):
        V_old = np.copy(V) #copy the previous value of V
        
        #go through all elements of V
        #system state
        for i_x1 in range(n_X):
            print(i_x1)
            for i_x2 in range(n_X):
                #find all nonzero probabilities in transition matrices T0 and T1
                #Start with T0
                nonzero_indices_T0 = np.flatnonzero(sys.T0[i_x1,i_x2,:,:])
                probability_values_T0 = np.ndarray.flatten(sys.T0[i_x1,i_x2,:,:])[nonzero_indices_T0]
                reward_0 = (sys.X_ss[i_x1]-sys.X_ss[i_x1]) ** 2
                #the same for T1
                nonzero_indices_T1 = np.flatnonzero(sys.T1[i_x1,i_x2,:,:])
                probability_values_T1 = np.ndarray.flatten(sys.T1[i_x1,i_x2,:,:])[nonzero_indices_T1]
                                
                #transmission energy values
                for i_T in range(n_T):
                    
                    #harvesting energy values
                    for i_H in range(n_H):
                        #find all nonzero probabilities in transition matrices T0 and T1
                        nonzero_indices_H = np.flatnonzero(sys.H10_distribution[i_H,:])
                        probability_values_H = np.ndarray.flatten(sys.H10_distribution[i_H,:])[nonzero_indices_H]
                        
                        #finally, go through battery values
                        for i_B in range(n_B):
                            
                            #find the minimum value of V and the corresponding action with given state
                            min_value, action = minimum_value(sys,i_B,i_T,discount_factor,balance_parameter,
                                                             nonzero_indices_T0,probability_values_T0,
                                                             nonzero_indices_T1,probability_values_T1,
                                                             reward_0,nonzero_indices_H,probability_values_H,V_old)
                            V[i_x1,i_x2,i_B,i_H,i_T] = min_value
                            
                            if N == N_steps - 1: #at the last step save the policy
                                policy[i_x1,i_x2,i_B,i_H,i_T] = action
    return policy
                                
def minimum_value(sys,current_index_B,current_index_T,discount_factor,balance_parameter,
                  indices_T0,values_T0,indices_T1,values_T1,reward_0,indices_H,values_H,V_old):             
 
    """
    An intermediate step in the value-iteration algorithm. Calculate the minimum expectation 
    of the value function V with given current state
        
    Parameters
    ----------
    
    sys: object of the class Model (the full state is: the dynamical system state X 
         the battery level, harvesting energy, transmission energy)
    current_index_B: the index of the current state of the battery
    current_index_T: the index of the current state of transmission energy
    discount_factor: the discount parameter in the algorithm
    balance_parameter: the balance parameter between the output penalty and the energy cost
    indices_T0: indices of nonzero elements of the transition matrix T0
    values_T0: nonzero elements of the transition matrix T0
    indices_T1: indices of nonzero elements of the transition matrix T1
    values_T1: nonzero elements of the transition matrix T1
    reward_0: basic part of the reward function
    indices_H: indices of nonzero elements of the transition matrix describing harvesting energy
    values_H: nonzero elements of the transition matrix describing harvesting energy
    V_old: the previous value of V

    Returns
    -------
    minimum value of V
    corresponding action (0 or 1)
    """
    V = [0,0] #initialize      
    for ind_T in range(len(sys.T_ss)):
        for ind_H in range(len(indices_H)):
            for action in range(1):
                if (action == 1) and (sys.B_ss[current_index_B] >= sys.T_ss[current_index_T]):
                    transmit = 1
                else:
                    transmit = 0
                
                #update battery level 
                B_next = sys.B_ss[current_index_B] + sys.H_ss[ind_H] + transmit * sys.H_ss[current_index_T]
                #find the correspondent value from battery state-space
                B_next_from_ss, ind_B_next_from_ss = match_value(B_next,sys.B_ss)
                #next calculate distribution of battery level
                if B_next >= B_next_from_ss:
                    if B_next >= sys.B_ss[-1]:
                        indices_B = [ind_B_next_from_ss]
                        values_B = [1]
                    else:
                        indices_B = [ind_B_next_from_ss,ind_B_next_from_ss + 1]
                        prob_temp = (B_next-B_next_from_ss)/(sys.B_ss[ind_B_next_from_ss + 1]-sys.B_ss[ind_B_next_from_ss])
                        values_B = [prob_temp, 1 - prob_temp]
                else:
                    if B_next <= sys.B_ss[0]:
                        indices_B = [ind_B_next_from_ss]
                        values_B = [1]
                    else:
                        indices_B = [ind_B_next_from_ss - 1,ind_B_next_from_ss]
                        prob_temp = (B_next_from_ss - B_next)/(sys.B_ss[ind_B_next_from_ss]-sys.B_ss[ind_B_next_from_ss - 1])
                        values_B = [1 - prob_temp, prob_temp]
                
                if action == 0:
                    indices_x = indices_T0
                    values_x = values_T0
                else:
                    indices_x = indices_T1
                    values_x = values_T1
                
                for ind_B in range(len(indices_B)):
                    reward = (1-transmit) * reward_0 + transmit * balance_parameter * (1-sys.B_ss[indices_B[ind_B]]/sys.B_ss[-1])
                    for ind_x in range(len(indices_x)):
                        ind_x1, ind_x2 = np.unravel_index(ind_x,(len(sys.X_ss),len(sys.X_ss)))
                        V[action] += values_x[ind_x] * values_B[ind_B] * values_H[ind_H] * sys.T_distribution[ind_T] *(reward
                                     + discount_factor * V_old[ind_x1, ind_x2, indices_B[ind_B], indices_H[ind_H],ind_T])
    if V[0] <= V[1]:
        return V[0], 0
    else:
        return V[1], 1
        