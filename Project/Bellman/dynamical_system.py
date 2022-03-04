import numpy as np
import math as mt
from .methods import *
import matplotlib.pyplot as plt

class Model:
    def __init__(self,sampling_step_h,x_initial,battery_initial,x_desired,recalculate=0):
        ''' Constructor for this class. '''
        self.X_ss = np.concatenate((np.arange(-15,-2.5,2),np.arange(-2.5,-0.9,0.5),np.arange(-0.9,1,0.1),np.arange(1,3,0.5),np.arange(3,8)),axis=0)
        self.h = sampling_step_h #sampling step, hours
        self.xd = x_desired
        self.x0 = np.array([[x_initial-x_desired], [0]])
        self.B0 = battery_initial
        a1 = 0.1
        a2 = 0.2 
        a = a1 + a2 
        k = -1.7
        a_h = mt.exp(-a * self.h)
        a_h_ = -1 / a * (mt.exp(-a * self.h) - 1) * k
        self.A1 = np.array([[a_h, a_h_],[0, 1]])
        self.A2 = np.array([[a_h_, -a_h_], [1, -1]])
    
        
        if recalculate:
            self.T0,  self.T1 = self.create_Transition_Matrix(system)
        else:
            self.T0 = np.load('T0.npy')
            self.T1 = np.load('T1.npy')
            
        self.H_ss = np.arange(0,61,10)
        self.H10_distribution = np.array([[0.0833,0.0833,0.833,0,0,0,0],
                               [0.25,0.375,0.25,0,0.125,0,0],
                               [0.19375,0,0.54375,0.0625,0.2,0,0],
                               [0,0,0.76923,0.076923,0.15385,0,0],
                               [0,0.0084,0.2521,0.0168,0.57143,0.06723,0.084],
                               [0,0,0,0,0.5,0.5,0],
                               [0,0,0,0,1,0,0]])
        self.B_ss = np.arange(0,1001,20)
        self.T_ss = np.arange(-25,-4,5)*3
        self.T_distribution = np.array([0.00348362203685777,0.0337897533569762,0.259234342176291,0.617973875960351,0.0855184064695248])
        
    def create_Transition_Matrix(self):
        #T0: with actions a = 0
        #T1: with actions a = 1
        n = len(self.X_ss)
        T0 = np.zeros((n,n,n,n))
        T1 = np.zeros((n,n,n,n))
  
        for N in range(50):
            print(N)
            for x1 in np.arange(self.X_ss[0]-1,self.X_ss[-1]+1,0.05):
                _, index1 = match_value(x1,self.X_ss)
                for x2 in np.arange(self.X_ss[0]-1,self.X_ss[-1]+1,0.05):
                    _, index2 = match_value(x2,self.X_ss)
                    
                    X = model_step(self,0,np.array([[x1],[x2]]))
                    _, index3 = match_value(X[0],self.X_ss)
                    _, index4 = match_value(X[1],self.X_ss)
                    T0[index1,index2,index3,index4] += 1
                    
                    X = model_step(self,1,np.array([[x1],[x2]]))
                    _, index3 = match_value(X[0],self.X_ss)
                    _, index4 = match_value(X[1],self.X_ss)
                    T1[index1,index2,index3,index4] += 1
        
        for index1 in range(n):
            for index2 in range(n):
                N = np.sum(np.sum(T0[index1,index2,:,:]))
                T0[index1,index2,:,:] /= N
    
                N = np.sum(np.sum(T1[index1,index2,:,:]))
                T1[index1,index2,:,:] /= N
        return T0, T1
    
    def simulate(self,policy,final_time):
        sampling_interval = np.arange(0,final_time+self.h,self.h)
        N = len(sampling_interval)
        X = np.zeros((2,N))
        Battery = np.zeros(N)
        Battery[0] = self.B0
        Harvested_Energy = np.zeros(N)
        Transmission_Energy = np.zeros(N)
        Transmission_Energy[0] = generate_sample(self.T_ss, self.T_distribution)
                
        X[:,0] = self.x0.T
        for i in range(0,N-1):
            _, index_x1 = match_value(X[0,i],self.X_ss)
            _, index_x2 = match_value(X[1,i],self.X_ss)
            _, index_B = match_value(Battery[i],self.B_ss)
            _, index_H = match_value(Harvested_Energy[i],self.H_ss)
            _, index_T = match_value(Transmission_Energy[i],self.H_ss)
            
            action = policy[index_x1,index_x2,index_B,index_H,index_T]
            if action == 1 and Battery[i] >= Transmission_Energy[i]:
                transmitted = 1
            else:
                transmitted = 0
            Harvested_Energy[i+1] = generate_sample(self.H_ss, self.H10_distribution[index_H,:])
            Transmission_Energy[i+1] = generate_sample(self.T_ss, self.T_distribution)
            if transmitted == 1:
               X[:,[i+1]] = mapping(X[:,[i-1]], self.X_ss, self.T1)
               Battery[i+1] = Battery[i] + Harvested_Energy[i+1]  + Transmission_Energy[i] # + due to  Transmission_Energy has negative values
            else:
               X[:,[i+1]] = mapping(X[:,[i-1]], self.X_ss, self.T0)
               Battery[i+1] = Battery[i] + Harvested_Energy[i+1]
            if Battery[i+1] > self.B_ss[-1]:
                Battery[i+1] = self.B_ss[-1]
            if Battery[i+1] < 0:
                Battery[i+1] = self.B_ss[0]
        return sampling_interval, X, Battery, Harvested_Energy, Transmission_Energy
               
               
    def simulate_with_random_policy(self,probability,final_time):
        sampling_interval = np.arange(0,final_time+self.h,self.h)
        N = len(sampling_interval)
        X = np.zeros((2,N))
        Battery = np.zeros(N)
        Battery[0] = self.B0
        Harvested_Energy = np.zeros(N)
        Transmission_Energy = np.zeros(N)
        Transmission_Energy[0] = generate_sample(self.T_ss, self.T_distribution)
                
        X[:,0] = self.x0.T
        for i in range(0,N-1):
            _, index_H = match_value(Harvested_Energy[i],self.H_ss)
            
            action = np.random.random() < probability
            if action and Battery[i] >= Transmission_Energy[i]:
                transmitted = 1
            else:
                transmitted = 0
            Harvested_Energy[i+1] = generate_sample(self.H_ss, self.H10_distribution[index_H,:])
            Transmission_Energy[i+1] = generate_sample(self.T_ss, self.T_distribution)
            if transmitted == 1:
               X[:,[i+1]] = mapping(X[:,[i-1]], self.X_ss, self.T1)
               Battery[i+1] = Battery[i] + Harvested_Energy[i+1]  + Transmission_Energy[i] # + due to  Transmission_Energy has negative values
            else:
               X[:,[i+1]] = mapping(X[:,[i-1]], self.X_ss, self.T0)
               Battery[i+1] = Battery[i] + Harvested_Energy[i+1]
            if Battery[i+1] > self.B_ss[-1]:
                Battery[i+1] = self.B_ss[-1]
            if Battery[i+1] < 0:
                Battery[i+1] = self.B_ss[0]
        return sampling_interval, X, Battery, Harvested_Energy, Transmission_Energy          
               
    def draw_temperature_battery(self,sampling_interval,X,Battery,title1 = '', title2 = '',xlabel1 = '', ylabel1 = '', xlabel2 = '', ylabel2 = ''):
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