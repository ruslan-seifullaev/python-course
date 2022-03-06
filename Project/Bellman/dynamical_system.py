import numpy as np
import math as mt
from .methods import *
import matplotlib.pyplot as plt

class Model:
    def __init__(self,sampling_step_h,x_initial,battery_initial,x_desired,recalculate=0):
        """
        Constructor for this class.
        Contains the full model consisting of:
        - dynamical system model (system matrices and state transition probability matrices)
        - battery model
        - energy harvesting model
        - transmission energy model     
        
        Parameters
        ----------
        
        sampling_step_h: step of sampling
        x_initial: initial state (temperature) value
        battery_initial: initial battery state
        x_desired: desired state
        recalculate: recalculate the model (it may take some time) or load pre-calculated on
                    from file (0 by default)

        Returns
        -------
        Model: an object of the class
        """
        #create dynamical system state space
        self.X_ss = np.concatenate((np.arange(-15,-2.5,2),np.arange(-2.5,-0.9,0.5),np.arange(-0.9,1,0.1),np.arange(1,3,0.5),np.arange(3,8)),axis=0)
        self.h = sampling_step_h #sampling step, hours
        self.xd = x_desired #desired state
        self.x0 = np.array([[x_initial-x_desired], [0]]) #initial state
        self.B0 = battery_initial #initial battery state
        self.A1, self.A2 = self.create_System_Matrices(sampling_step_h,0.1,0.2,-1.7)
       
        if recalculate:
            #calculate transition probability matrices
            self.T0,  self.T1 = self.create_Transition_Matrix(system)
        else:
            #load the matrices from file
            self.T0 = np.load('T0.npy')
            self.T1 = np.load('T1.npy')
            
        self.H_ss = np.arange(0,61,10) #harvesting energy state space
        #harvesting energy transition matrix (distribution)
        self.H10_distribution = np.array([[0.0833,0.0833,0.833,0,0,0,0],
                               [0.25,0.375,0.25,0,0.125,0,0],
                               [0.19375,0,0.54375,0.0625,0.2,0,0],
                               [0,0,0.76923,0.076923,0.15385,0,0],
                               [0,0.0084,0.2521,0.0168,0.57143,0.06723,0.084],
                               [0,0,0,0,0.5,0.5,0],
                               [0,0,0,0,1,0,0]])
        self.B_ss = np.arange(0,1001,20) #battery level state space
        self.T_ss = np.arange(-25,-4,5)*3 #transmission energy state space
        #transmission energy distribution
        self.T_distribution = np.array([0.00348362203685777,0.0337897533569762,0.259234342176291,0.617973875960351,0.0855184064695248])
        
    def create_System_Matrices(self,sampling_step_h,wall_conductivity,ground_conductivity,control_gain):
        """
        Create system matrices A1 and A2 in the sampled-data temperature control model 
        
        Parameters
        ----------
        
        sampling_step_h: step of sampling
        wall_conductivity: wall conductivity 
        ground_conductivity: ground conductivity
        control_gain: control gain scalar

        Returns
        -------
        A1, A2: system matrices A1 and A2
        """
        total_conductivity = wall_conductivity + ground_conductivity 
        component1 = mt.exp(-total_conductivity * sampling_step_h)
        component2 = -1 / total_conductivity * (mt.exp(-total_conductivity * sampling_step_h) - 1) * control_gain
        A1 = np.array([[component1, component2],[0, 1]])
        A2 = np.array([[component2, -component2], [1, -1]])
        return A1, A2
        
    def create_Transition_Matrix(self):
        """
        Create transition probability matrices for stochastic approximation]
        
        Parameters
        ----------

        Returns
        -------
        T0: transition probability matrix with 'action=0'
        T1: transition probability matrix with 'action=1'
        """
        n = len(self.X_ss)
        T0 = np.zeros((n,n,n,n))
        T1 = np.zeros((n,n,n,n))
  
        for N in range(50):
            print(N)
            #go through all possible states of the system
            for x1 in np.arange(self.X_ss[0]-1,self.X_ss[-1]+1,0.05):
                _, index1 = match_value(x1,self.X_ss)
                for x2 in np.arange(self.X_ss[0]-1,self.X_ss[-1]+1,0.05):
                    _, index2 = match_value(x2,self.X_ss) #find indices of the point X_n=[x1,x2]
                    
                    #do with action=0
                    X = model_step(self,0,np.array([[x1],[x2]])) #generate next step X_{n+1}
                    _, index3 = match_value(X[0],self.X_ss) #find indices of the point X_{n+1}
                    _, index4 = match_value(X[1],self.X_ss)
                    T0[index1,index2,index3,index4] += 1 #increase corresponding element of T0
                    
                    #do with action=1
                    X = model_step(self,1,np.array([[x1],[x2]])) #generate next step X_{n+1}
                    _, index3 = match_value(X[0],self.X_ss) #find indices of the point X_{n+1}
                    _, index4 = match_value(X[1],self.X_ss)
                    T1[index1,index2,index3,index4] += 1 #increase corresponding element of T1
        
        #normalize obtained matrices to get probabilities
        for index1 in range(n):
            for index2 in range(n):
                N = np.sum(np.sum(T0[index1,index2,:,:]))
                T0[index1,index2,:,:] /= N
    
                N = np.sum(np.sum(T1[index1,index2,:,:]))
                T1[index1,index2,:,:] /= N
        return T0, T1
    
    def simulate(self,policy,final_time):
        """
        Simulate the MDP model for t = 0:final_time. Return the resulting trajectories
        
        Parameters
        ----------
        policy: ndarray with transmission policy
        final_time: simulation stop time (hours)

        Returns
        -------
        sampling_interval: interval of sampling times
        X: state trajectory (temperature)
        Battery: battery level values
        Harvested_Energy: harvesting energy values
        Transmission_Energy: transmission energy values
        """
        #initialization
        sampling_interval = np.arange(0,final_time+self.h,self.h)
        N = len(sampling_interval)
        X = np.zeros((2,N))
        Battery = np.zeros(N)
        Battery[0] = self.B0
        Harvested_Energy = np.zeros(N)
        Transmission_Energy = np.zeros(N)
        Transmission_Energy[0] = generate_sample(self.T_ss, self.T_distribution)
                
        X[:,0] = self.x0.T
        for i in range(0,N-1): #go through all times and generate samples
            #find current state indices in state spaces
            _, index_x1 = match_value(X[0,i],self.X_ss)
            _, index_x2 = match_value(X[1,i],self.X_ss)
            _, index_B = match_value(Battery[i],self.B_ss)
            _, index_H = match_value(Harvested_Energy[i],self.H_ss)
            _, index_T = match_value(Transmission_Energy[i],self.H_ss)
            
            #define current action according to the policy
            action = policy[index_x1,index_x2,index_B,index_H,index_T] 
            #check if we have enough energy to perform transmission
            if action == 1 and Battery[i] >= Transmission_Energy[i]:
                transmitted = 1
            else:
                transmitted = 0
            #generate a sample of harvesting energy
            Harvested_Energy[i+1] = generate_sample(self.H_ss, self.H10_distribution[index_H,:])
            #generate a sample of transmission energy
            Transmission_Energy[i+1] = generate_sample(self.T_ss, self.T_distribution)
            if transmitted == 1:
               #generate a sample of the system state with 'action=1'
               X[:,[i+1]] = mapping(X[:,[i-1]], self.X_ss, self.T1)
               #calculate the new battery state
               Battery[i+1] = Battery[i] + Harvested_Energy[i+1]  + Transmission_Energy[i] # + due to  Transmission_Energy has negative values
            else:
               #generate a sample of the system state with 'action=0'
               X[:,[i+1]] = mapping(X[:,[i-1]], self.X_ss, self.T0)
               #calculate the new battery state
               Battery[i+1] = Battery[i] + Harvested_Energy[i+1]
            if Battery[i+1] > self.B_ss[-1]: #full battery
                Battery[i+1] = self.B_ss[-1]
            if Battery[i+1] < 0: #empty battery
                Battery[i+1] = self.B_ss[0]
        return sampling_interval, X, Battery, Harvested_Energy, Transmission_Energy
               
               
    def simulate_with_random_policy(self,probability,final_time):
        """
        Simulate the MDP model for t = 0:final_time with random policy. Return the resulting trajectories
        
        Parameters
        ----------
        probability: probability of transmission
        final_time: simulation stop time (hours)

        Returns
        -------
        sampling_interval: interval of sampling times
        X: state trajectory (temperature)
        Battery: battery level values
        Harvested_Energy: harvesting energy values
        Transmission_Energy: transmission energy values
        """
        #initialization
        sampling_interval = np.arange(0,final_time+self.h,self.h)
        N = len(sampling_interval)
        X = np.zeros((2,N))
        Battery = np.zeros(N)
        Battery[0] = self.B0
        Harvested_Energy = np.zeros(N)
        Transmission_Energy = np.zeros(N)
        Transmission_Energy[0] = generate_sample(self.T_ss, self.T_distribution)
                
        X[:,0] = self.x0.T
        for i in range(0,N-1):#go through all times and generate samples
            _, index_H = match_value(Harvested_Energy[i],self.H_ss)
            
            #generate a random action according to given probability
            action = np.random.random() < probability
            #check if we have enough energy to perform transmission
            if action and Battery[i] >= Transmission_Energy[i]:
                transmitted = 1
            else:
                transmitted = 0
            #generate a sample of harvesting energy
            Harvested_Energy[i+1] = generate_sample(self.H_ss, self.H10_distribution[index_H,:])
            #generate a sample of transmission energy
            Transmission_Energy[i+1] = generate_sample(self.T_ss, self.T_distribution)
            if transmitted == 1:
               #generate a sample of the system state with 'action=1'
               X[:,[i+1]] = mapping(X[:,[i-1]], self.X_ss, self.T1)
               #calculate the new battery state
               Battery[i+1] = Battery[i] + Harvested_Energy[i+1]  + Transmission_Energy[i] # + due to  Transmission_Energy has negative values
            else:
               #generate a sample of the system state with 'action=0'
               X[:,[i+1]] = mapping(X[:,[i-1]], self.X_ss, self.T0)
               #calculate the new battery state
               Battery[i+1] = Battery[i] + Harvested_Energy[i+1]
            if Battery[i+1] > self.B_ss[-1]: #full battery
                Battery[i+1] = self.B_ss[-1]
            if Battery[i+1] < 0: #empty battery
                Battery[i+1] = self.B_ss[0]
        return sampling_interval, X, Battery, Harvested_Energy, Transmission_Energy          
               
    def draw_temperature_battery(self,sampling_interval,X,Battery,title1 = '', title2 = '',xlabel1 = '', ylabel1 = '', xlabel2 = '', ylabel2 = ''):
        """
        Create a figure with two subplots illustrating the state (temperature) and battery trajectories
        
        Parameters
        ----------
        sampling_interval: interval of sampling times
        X: state trajectory (temperature)
        Battery: battery level values
        title1: title of the first subplot, default('')
        title2: title of the second subplot, default('')
        xlabel1: label of the x-axis in the first subplot, default('')
        ylabel1: label of the y-axis in the first subplot, default('')
        xlabel2: label of the x-axis in the second subplot, default('')
        ylabel2: label of the y-axis in the second subplot, default('')
 
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        ax1.set_title(title1)
        ax1.plot(sampling_interval,X[0,:].T+self.xd)
        ax1.set(xlim=(0, sampling_interval[-1]))
        ax1.set_xlabel(xlabel1)
        ax1.set_ylabel(ylabel1)
        
        ax2.set_title(title2)
        ax2.plot(sampling_interval,Battery)
        ax2.set(xlim=(0, sampling_interval[-1]))
        ax2.set_xlabel(xlabel2)
        ax2.set_ylabel(ylabel2)       
        
  
        plt.show()     