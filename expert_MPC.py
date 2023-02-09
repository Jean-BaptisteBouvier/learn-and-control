# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:39:56 2023

@author: Jean-Baptiste Bouvier

Functions to generate expert trajectories with Model Predictive Control (MPC).
At each state, proceed with random shooting and gets reward for each random trajectory.
We only apply the first action generating the highest reward, before reusing random shooting at the next state.

The reward is quadratic and inverse quadratic in the state to ensure convergence to 0.
Matrix M parametrizes the reward, and hence which states must be driven to 0 first.
Scalar H is the random shooting horizon.
Scalar N_rep is the number of time each action is repeated to increase the outcome difference between actions.
Scalar N_traj_shooting is the number of random shooting trajectories sampled.
"""



import copy
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset




# Reward function for the expert controller
def reward(s, a, M):
    """Reward function for the MPC random shooting.
    Takes in the current state 's' and a given action 'a'.
    Output is a scalar reward value, the higher the better."""
    p = s @ M @ s.t()
    return -torch.norm(a) + 1./p - p


# MPC with random shooting to determine the best action to take at each state
def best_action(sys, dynamics, s0, N_traj, H, N_rep, M):
    """Calculates the best action to take at a given state 's' using the
    random shooting method of testing 'N_traj' different trajectories
    of 'H' steps each, i.e., 'H' random actions and comparing their overall reward."""
   
    a_sz = sys.action_size
    best_reward = -10**10
    for traj_id in range(N_traj):
        
        first_a = (sys.input_max-sys.input_min)*torch.rand((1,a_sz)) + sys.input_min # uniform action sampling
        r = 0
        s = copy.deepcopy(s0) # with s = s0, s0 is modified when s is
        a = first_a
       
        with torch.no_grad():
            for step in range(H):
                # repeat same action several times for faster exploration
                for repeat in range(N_rep):
                    s = dynamics.next_state(s, a)
                    r += reward(s, a, M) # add reward of past action and current state
                
                a = (sys.input_max-sys.input_min)*torch.rand((1,a_sz)) + sys.input_min
        
        if r > best_reward:
            best_a = first_a
            best_reward = r
 
    return best_a
   
   
# Expert data aggregation using MPC
def expert_data_aggregation(sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M):
    """Generate a number 'N_traj_aggreg' trajectories of length 'N_pred'
    starting from random initial states propagated through the best action
    random shooting method. 'M' is a positive definite matrix weighing the state reward."""
   
    inputs = torch.zeros((N_pred*N_traj_aggreg, sys.state_size + sys.action_size))
    labels = torch.zeros((N_pred*N_traj_aggreg, sys.state_size))
    i = 0
    
    for traj_id in range(N_traj_aggreg):
        s = 0.2*torch.randn((1,sys.state_size)) # normal distribution of initial states
        
        for t in range(N_pred):
            a = best_action(sys, dynamics, s, N_traj_shooting, H, N_rep, M)
            inputs[i] = torch.cat((s, a), dim=1)
            labels[i] = dynamics.next_state_dif(s, a)
            s += labels[i]
            i += 1
            if torch.norm(inputs[i-1,:sys.state_size]) < 0.01: # converged trajectory
                break
            
        phase_portrait(inputs[i-t-1:i, :sys.state_size], sys, str(traj_id+1)+'/'+str(N_traj_aggreg))
        # plot_actions(inputs)
    
    return inputs[:i], labels[:i]


    
# Class for the dataset of expert data points collected with MPC
class Expert_Dataset(Dataset):
    def __init__(self, sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M):   
        [self.X, self.y] = expert_data_aggregation(sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M)
        self.N = self.X.shape[0]
       
    def __len__(self):# number of data points
        return self.N
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]

    # adds more on-policy data points to the dataset
    def add_data(self, sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M):
        [inputs, labels] = expert_data_aggregation(sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M)
        self.N += inputs.shape[0]
        self.X = torch.cat((self.X, inputs), dim=0)
        self.y = torch.cat((self.y, labels), dim=0)



# Plots a phase portrait of the input data 
def phase_portrait(inputs, sys, extra_title_label = ''):
    """Plot phase portrait of a given input set.
    Only works for systems with an even number states composed of (variables, their derivative), 
    for instance (theta, theta_dot), or (x, y, z, x_dot, y_dot, z_dot).
    The 'inputs' set should contain trajectories of these states in the order described above."""
    
    n = int(sys.state_size/2) # number of states of the same derivative degree
    plt.title('Phase portrait ' + extra_title_label)
    
    plt.scatter(inputs[0,0], inputs[0,n], s=20, c='g', label="start", zorder=2)
    plt.scatter(inputs[-1,0], inputs[-1,n], s=20, c='r', label="end", zorder=2)
    for i in range(n):
        plt.plot(inputs[:,i].detach().numpy(), inputs[:,i+n].detach().numpy(), label=sys.state_labels[i])
        plt.scatter(inputs[0,i], inputs[0,i+n], s=20, c='g', zorder=2)
        plt.scatter(inputs[-1,i], inputs[-1,i+n], s=20, c='r', zorder=2)
    
    plt.xlabel('angle')
    plt.ylabel('angular velocity')
    plt.legend()
    plt.show()
   
    
   
# Plots the actions taken in a given input set
def plot_actions(inputs):
    """Plots the actions taken in a given input set"""
    plt.title('Actions') 
    plt.plot(np.arange(inputs.shape[0]), inputs[:,2].detach().numpy())  
    plt.ylabel('a')
    plt.show()
    


# Querying expert MPC controller on the states visited by the on-policy rollout
def MPC_query(sys, dynamics, states, N_traj_shooting, H, N_rep, M):
    """Given a tensor of states, asks the expert controller (MPC) to predict the best actions
    to take at each of these states."""
    N = states.shape[0]
    actions = torch.zeros((N, sys.action_size))
    
    for i in range(N):
        actions[i] = best_action(sys, dynamics, states[i].unsqueeze(dim=0), N_traj_shooting, H, N_rep, M) # MPC
    
    return states, actions
