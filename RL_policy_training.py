# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:39:56 2023

@author: jeanb

Functions to perform the Reinforcement Learning of a good controller.
Based on this controller a NN learns a policy to generalize this controller.
"""



import copy
import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dynamics_learning import train_loop



# Reward function for RL
def reward(s, a, M):
    """Reward function for the model-based Reinforcement Learning part.
    Takes in the current state 's' and a given action 'a'.
    Output is a scalar reward value, the higher the better"""
    p = s @ M @ s.t()
    return -torch.norm(a) + 1./p - p


# Random shooting method to determine the best action to take
def best_action(sys, dynamics, s0, N_traj, H, N_rep, M):
    """Calculates the best action to take at a given state 's' using the
    random shooting method of testing 'N_traj' different trajectories
    of 'H' steps each, i.e., 'H' random actions and comparing their overall reward."""
   
    
    # plt.title('Shooting') 
    # plt.scatter(s0[0,0], s0[0,1], s=20, c='g', label="start", zorder=2)
    
    a_sz = sys.action_size
    
    best_reward = -10**10
    for traj_id in range(N_traj):
        
        first_a = (sys.input_max-sys.input_min)*torch.rand((1,a_sz)) + sys.input_min # uniform sampling
        r = 0
        s = copy.deepcopy(s0) # with s = s0, s0 is modified when s is
        a = first_a
        # traj = torch.zeros((H, s_sz))
        # traj[0] = s0
       
        with torch.no_grad():
            for step in range(H):
                # repeat same action several times for faster exploration
                for repeat in range(N_rep):
                    s = dynamics.next_state(s, a)
                    r += reward(s, a, M) # add reward of past action and current state
                
                a = (sys.input_max-sys.input_min)*torch.rand((1,a_sz)) + sys.input_min
                
                # traj[step+1] = s
        
        # plt.plot(traj[:,0].detach().numpy(), traj[:,1].detach().numpy())
        # plt.text(traj[-1,0].detach().numpy(), traj[-1,1].detach().numpy(), str(round(r.item()*10)/10))
        
        if r > best_reward:
            best_a = first_a
            best_reward = r
         
    # plt.xlabel('theta')
    # plt.ylabel('theta dot')
    # plt.show()
            
    return best_a
   
   
# On-policy data aggregation
def on_policy_data(sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M):
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
            
        phase_portrait(inputs[i-t-1:i, :sys.state_size], sys)
        
    # plot_actions(inputs)
    return inputs[:i], labels[:i]


    
# Class for the dataset of on-policy data points collected with function on_policy_data
class OnPolicy_Dataset(Dataset):
    def __init__(self, sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M):   
        [self.X, self.y] = on_policy_data(sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M)
        self.N = self.X.shape[0]
       
    def __len__(self):# number of data points
        return self.N
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]

    # adds more on-policy data points to the dataset
    def add_data(self, sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M):
        [inputs, labels] = on_policy_data(sys, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, M)
        self.N += inputs.shape[0]
        self.X = torch.cat((self.X, inputs), dim=0)
        self.y = torch.cat((self.y, labels), dim=0)



# Plots actual trajectory and the learned ones 
def phase_portrait(inputs, sys, extra_title_label = ''):
    """Plot phase portrait of given data"""
    
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
    

    

# Neural Network class for the policy to be learned from the RL trajectories
class Policy_NN(nn.Module):
    def __init__(self, sys, num_hiddens = 32):
        super().__init__()
        num_inputs = sys.state_size
        num_outputs = sys.action_size
        self.net = nn.Sequential(nn.Linear(num_inputs, num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(num_outputs))    
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        return self.net(x)
    
    def action(self, s):
        """Returns the action to take predicted by the policy at given state 's'."""
        with torch.no_grad():
            return self.net(s) # policy training data is not normalized


# Dataset class for the policy training consisting of states as inputs
# and action to be taken at given step as labels
class Policy_Data(Dataset):
    def __init__(self, sys, on_policy_dataset):
        
        self.state_size = sys.state_size
        self.action_size = sys.action_size
        self.X = on_policy_dataset.X[:,:self.state_size] # states
        self.y = on_policy_dataset.X[:,self.state_size:].unsqueeze(dim=1) # corresponding action in 1 column
        self.N = on_policy_dataset.N
       
    def __len__(self):# number of data points
        return self.N
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]
    
    def add_data(self, inputs, labels):
        self.X = torch.cat((self.X, inputs), dim=0)
        self.y = torch.cat((self.y, labels.unsqueeze(dim=1)), dim=0)
        self.N += inputs.shape[0]
    

# Network policy training and plotting training and testing loss
def policy_training(policy_data, policy, epochs=30, lr=0.001, batch_size = 2**6):
    """Trains the policy network on the training dataset."""
    train_dataloader = DataLoader(policy_data, batch_size, shuffle=True)
    optimizer = torch.optim.SGD(policy.parameters(), lr)
    
    training_loss = np.zeros(epochs)
    
    plot_window = 50
    writing_window = 10
    
    for t in range(epochs):
        training_loss[t] = train_loop(train_dataloader, policy, optimizer)

        optimizer = torch.optim.SGD(policy.parameters(), lr/(1+t)**0.5, momentum=0.5) # squarely decreasing lr
        
        if t%writing_window == 0:
            print(f'epoch: {t:3}  training loss: {training_loss[t]:10.8f}')    
        
        if t > 0 and t%plot_window == 0:
            plt.title('Policy training: loss')
            plt.plot(np.arange(t-plot_window, t), training_loss[t-plot_window:t], 'b')
            plt.xlabel('epochs')
            plt.show()
      
    plt.title('Policy training: loss')
    plt.plot(np.arange(epochs), training_loss, 'b')
    plt.xlabel('epochs')
    plt.show()
    print(f"Training loss stabilizing around {training_loss[round(epochs/2):].mean():.8f}")
    return training_loss



# On-policy rollout, i.e., trajectory propagations
def on_policy_rollout(sys, dynamics, policy, N_pred, N_traj):
    """Generate a number 'N_traj' of trajectories of length 'N_pred'
    starting from random initial states propagated through the policy and dynamics
    neural networks."""
   
    states = torch.zeros((N_pred*N_traj+1, sys.state_size))
    i = 0
    
    for traj_id in range(N_traj):
        states[i] = 0.2*torch.randn((1, sys.state_size)) # normal distribution of initial states
   
        for t in range(N_pred):
            states[i+1] = dynamics.next_state(states[i], policy.action(states[i]))
            i += 1
            # if torch.norm(states[i]) < 0.01: # converged trajectory
            #     break
            
        phase_portrait(states[i-t-1:i+1], sys)
    
    return states[:i]


# Querying 'expert' MPC controller on the states visited by the on-policy rollout
def MPC_query(sys, dynamics, states, N_traj_shooting, H, N_rep, M):
    N = states.shape[0]
    actions = torch.zeros((N, sys.action_size))
    
    for i in range(N):
        actions[i] = best_action(sys, dynamics, states[i].unsqueeze(dim=0), N_traj_shooting, H, N_rep, M) # MPC
    
    return states, actions
