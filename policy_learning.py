# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:46:27 2023

@author: Jean-Baptiste Bouvier

Functions to learn a policy based on a set of expert data.
The expert dataset should contain states and the associated action to take.
"""


import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from expert_MPC import phase_portrait
from dynamics_learning import train_loop




# Neural Network class for the policy to be learned from the expert trajectories
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
class Policy_Dataset(Dataset):
    def __init__(self, sys, on_policy_dataset):
        
        self.state_size = sys.state_size
        self.action_size = sys.action_size
        self.X = on_policy_dataset.X[:,:self.state_size] # states
        self.y = on_policy_dataset.X[:,self.state_size:] # corresponding action in 1 column
        self.N = on_policy_dataset.N
       
    def __len__(self):# number of data points
        return self.N
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]
    
    def add_data(self, inputs, labels):
        self.X = torch.cat((self.X, inputs), dim=0)
        self.y = torch.cat((self.y, labels), dim=0)
        self.N += inputs.shape[0]
    

# Network policy training and plotting training and testing loss
def policy_training(policy_data, policy, epochs=30, lr=0.001, batch_size = 2**6):
    """Trains the policy network on the training dataset."""
    train_dataloader = DataLoader(policy_data, batch_size, shuffle=True)
    optimizer = torch.optim.SGD(policy.parameters(), lr)
    
    training_loss = np.zeros(epochs)
    
    writing_window = 10
    
    for t in range(epochs):
        training_loss[t] = train_loop(train_dataloader, policy, optimizer)

        optimizer = torch.optim.SGD(policy.parameters(), lr/(1+t)**0.5, momentum=0.5) # squarely decreasing lr
        
        if t%writing_window == 0:
            print(f'epoch: {t:3}  training loss: {training_loss[t]:10.8f}')
      
    plt.title('Policy training: loss')
    plt.plot(np.arange(epochs), training_loss, 'b')
    plt.xlabel('epochs')
    plt.show()
    print(f"Training loss stabilizing around {training_loss[round(epochs/2):].mean():.8f}")
    return training_loss



# On-policy rollout, i.e., trajectory propagations
def on_policy_rollout(sys, dynamics, policy, N_pred, N_traj):
    """Generates a number 'N_traj' of trajectories of length 'N_pred'
    starting from random initial states propagated through the policy and dynamics
    neural networks."""
   
    states = torch.zeros((N_pred*N_traj+1, sys.state_size))
    i = 0
    
    for traj_id in range(N_traj):
        states[i] = 0.2*torch.randn((1, sys.state_size)) # normal distribution of initial states
   
        for t in range(N_pred):
            states[i+1] = dynamics.next_state(states[i], policy.action(states[i]))
            i += 1
            
        phase_portrait(states[i-t-1:i+1], sys)
    
    return states[:i]

