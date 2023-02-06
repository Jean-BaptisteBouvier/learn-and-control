# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:51:45 2023

@author: jeanb

Build predictors that take a state and decide which action to take and what the resulting state will be.
Model-free learning........ Doesn't work
Instead, build a combined NN that puts together dynamics_NN and policy_NN on top of each other into one single NN.
"""


import torch
from torch import nn
from torch.utils.data import Dataset




# Neural Network class for the predictor
# Takes a state 's(t)' as input and its output is 
# the next state difference 's(t+dt)-s(t)' after taking an action 'a(t)'
class Predictor_NN(nn.Module):
    def __init__(self, sys, num_hiddens = 64):
        super().__init__()
        
        self.state_size = sys.state_size
        self.action_size = sys.action_size
        self.net = nn.Sequential(nn.Linear(self.state_size, num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(self.state_size + self.action_size))    
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        return self.net(x)
    
    def take_step(self, s):
        """Returns the next state difference and the action."""
        with torch.no_grad():
            return self.net(s) # data is not normalized
        
    def next_state_dif_action(self, s):
        """Returns a pair (next state difference, action)."""
        y = self.take_step(s)
        return y[:self.state_size], y[self.state_size:] # returns a pair next state difference, action

    def next_state_action(self, s):
        """Returns a pair (next state, action)."""
        y = self.take_step(s)
        return y[:self.state_size] + s, y[self.state_size:] # returns a pair next state, action

    def next_action(self, s):
        """Returns the action."""
        y = self.take_step(s)
        return y[-self.action_size:]
    
    

# Creates the data to train the NN controlled_pendulum
def ctrl_data(policy, dynamics, N):
    inputs = torch.zeros((N, dynamics.state_size))
    labels = torch.zeros((N, dynamics.state_size + dynamics.action_size))
    
    for i in range(N):
        inputs[i] = 0.2*torch.randn((1, dynamics.state_size)) # normal distribution of initial states
        a = policy.action(inputs[i])
        ds = dynamics.next_state_dif(inputs[i], a)
        labels[i] = torch.cat((ds, a), dim=0)
    
    return inputs, labels



# Dataset class for the training of the controlled pendulum consisting of states as inputs
# and labels as next state difference + action 
class Predictor_Dataset(Dataset):
    def __init__(self, policy, dynamics, N):   
        self.X, self.y = ctrl_data(policy, dynamics, N)
        self.N = N
       
    def __len__(self):# number of data points
        return self.N
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]
    
    def add_data(self, inputs, labels):
        self.X = torch.cat((self.X, inputs), dim=0)
        self.y = torch.cat((self.y, labels), dim=0)
        self.N += inputs.shape[0]

################################# TO DO #######################################
# Implement a better algorithm for the imitation model-free learning:
# TRPO (Trust Region Policy Optimization)
# or PPO (Proximal Policy Optimization)

###############################################################################
















### Builds the combined neural network of policy and dynamics
class combined_NN(nn.Module):
    """Given the policy and dynamics NN, we combine them into one single NN
    Input: state           Output: next state dif, action
    Applies policy on the state while conserving the state unchanged in parallel,
    then applies dynamics on the (state, action) pair, while conserving the action unchanged in parallel."""
    def __init__(self, dynamics, policy):
        super(combined_NN, self).__init__()
        
        self.state_size = dynamics.state_size
        self.action_size = dynamics.action_size
        
        self.num_inputs = self.state_size # state
        self.num_outputs = self.state_size + self.action_size # next state dif, action
        self.net = nn.Sequential()
        
        # Build the first stage of the NN by copying policy and keeping the state in parallel
        # Because of the ReLu layers, to conserve each state s unchanged,
        # we build  s_plus = s  and  s_minus = -s
        num_hidden_policy = policy.net[0].out_features
        assert(policy.net[-1].out_features == self.action_size)
        
        # Build a weight matrix W to split the states into + and -
        W = torch.zeros((2*self.state_size, self.state_size))
        for i in range(self.state_size):
            W[2*i, i] = 1
            W[2*i+1, i] = -1
        
        for layer in policy.net:
            if(isinstance(layer, nn.Linear)):
                with torch.no_grad():
                    if layer.weight.shape[1] == self.num_inputs: # first layer of policy_NN
                        new_layer = nn.Linear(self.num_inputs, num_hidden_policy + 2*self.num_inputs)
                        new_layer.weight.copy_(torch.cat( (W, layer.weight), dim=0)) 
                        new_layer.bias.copy_(torch.cat( (torch.zeros(2*self.num_inputs), layer.bias), dim=0))
                
                    elif layer.weight.shape[0] == self.action_size: # last layer of policy_NN
                        new_layer = nn.Linear(num_hidden_policy + 2*self.num_inputs, self.num_outputs)
                        W_11 = W.t() # matrix to rebuild the states based on their ReLU invariants
                        W_12 = torch.zeros((self.num_inputs, num_hidden_policy))
                        W_21 = torch.zeros((self.action_size, 2*self.num_inputs))
                        W_22 = layer.weight
                        new_layer.weight.copy_(torch.cat( (torch.cat((W_11, W_12), dim=1), torch.cat((W_21, W_22), dim=1)), dim=0)) 
                        new_layer.bias.copy_(torch.cat( (torch.zeros(self.num_inputs), layer.bias), dim=0))
                
                    else: # hidden layer of policy_NN
                        new_layer = nn.Linear(num_hidden_policy + 2*self.num_inputs, num_hidden_policy + 2*self.num_inputs)
                        W_11 = torch.eye(2*self.num_inputs)
                        W_12 = torch.zeros((2*self.num_inputs, num_hidden_policy))
                        W_21 = torch.zeros((num_hidden_policy, 2*self.num_inputs))
                        W_22 = layer.weight
                        new_layer.weight.copy_(torch.cat( (torch.cat((W_11, W_12), dim=1), torch.cat((W_21, W_22), dim=1)), dim=0))
                        new_layer.bias.copy_(torch.cat( (torch.zeros(2*self.num_inputs), layer.bias), dim=0))
                
                    self.net.append(new_layer)
            if(isinstance(layer, nn.ReLU)):
                self.net.append( nn.ReLU() )
            if 'Flatten' in (str(layer.__class__.__name__)): 
                continue # don't add the Flatten layer
                
        # Build the second stage of the NN by copying dynamics and keeping the action in parallel
        # Because of the ReLu layers, we build  a_plus = a  and  a_minus = -a.
        
        # Normalization layer for the dynamics with outputs: normalized state, normalized action, and regular state
        new_layer = nn.Linear(self.num_outputs, self.state_size + 2*self.action_size)
        with torch.no_grad():
            W_1 = torch.diag(1./dynamics.input_std)
            W_21 = torch.zeros((self.action_size, self.state_size))
            W_22 = torch.eye(self.action_size)
            new_layer.weight.copy_( torch.cat( (W_1, torch.cat((W_21, W_22), dim=1)), dim=0) )
            new_layer.bias.copy_( torch.cat( (-dynamics.input_mean/dynamics.input_std, torch.zeros(self.action_size)) ))
            self.net.append(new_layer)
        
        # Build a weight matrix W to split the actions into + and -
        W = torch.zeros((2*self.action_size, self.action_size))
        for i in range(self.action_size):
            W[2*i, i] = 1
            W[2*i+1, i] = -1
        
        num_hidden_dyna = dynamics.net[0].out_features
        for layer in dynamics.net:
            if(isinstance(layer, nn.Linear)):
                with torch.no_grad():
                    if layer.weight.shape[1] == self.state_size + self.action_size: # first layer of dynamics_NN
                        new_layer = nn.Linear(self.state_size + 2*self.action_size, num_hidden_dyna + 2*self.action_size)
                        W_11 = layer.weight
                        W_12 = torch.zeros((num_hidden_dyna, self.action_size))
                        W_21 = torch.zeros((2*self.action_size, self.num_outputs))
                        W_22 = W
                        new_layer.weight.copy_(torch.cat( (torch.cat((W_11, W_12), dim=1), torch.cat((W_21, W_22), dim=1)), dim=0)) 
                        new_layer.bias.copy_(torch.cat( (layer.bias, torch.zeros(2*self.action_size)), dim=0))
                
                    elif layer.weight.shape[0] == self.state_size: # last layer of dynamics_NN
                        new_layer = nn.Linear(num_hidden_dyna + 2*self.action_size, self.state_size+self.action_size)
                        W_11 = layer.weight
                        W_12 = torch.zeros((self.state_size, 2*self.action_size))
                        W_21 = torch.zeros((self.action_size, num_hidden_dyna))
                        W_22 = W.t() # matrix to rebuild action based on its ReLU invariants
                        new_layer.weight.copy_(torch.cat( (torch.cat((W_11, W_12), dim=1), torch.cat((W_21, W_22), dim=1)), dim=0)) 
                        new_layer.bias.copy_(torch.cat( (layer.bias, torch.zeros(self.action_size)), dim=0))
                
                    else: # hidden layers of dynamics_NN
                        new_layer = nn.Linear(num_hidden_dyna + 2*self.action_size, num_hidden_dyna + 2*self.action_size)
                        W_11 = layer.weight
                        W_12 = torch.zeros((num_hidden_dyna, 2*self.action_size))
                        W_21 = torch.zeros((2*self.action_size, num_hidden_dyna))
                        W_22 = torch.eye(2*self.action_size)
                        new_layer.weight.copy_(torch.cat( (torch.cat((W_11, W_12), dim=1), torch.cat((W_21, W_22), dim=1)), dim=0))
                        new_layer.bias.copy_(torch.cat( (layer.bias, torch.zeros(2*self.action_size)), dim=0))
                
                    self.net.append(new_layer)
            if(isinstance(layer, nn.ReLU)):
                self.net.append( nn.ReLU() )
            if 'Flatten' in (str(layer.__class__.__name__)): 
                continue # don't add the Flatten layer
        
        # De-normalization layer
        self.net.append(nn.Linear(self.num_outputs, self.num_outputs))
        with torch.no_grad():
            self.net[-1].weight.copy_( torch.diag(torch.cat( (dynamics.output_std, torch.ones(self.action_size)) ) ) )
            self.net[-1].bias.copy_( torch.cat( (dynamics.output_mean, torch.zeros(self.action_size)), dim=0) )
        
            
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        return self.net(x)
    
    def take_step(self, s):
        """Returns the action to take predicted by the policy at given state 's'
        and the next state difference."""
        with torch.no_grad():
            return self.net(s).squeeze() # data is not normalized
        
    def next_state_dif_action(self, s):
        """Returns a pair (next state difference, action)."""
        y = self.take_step(s)
        return y[:self.state_size], y[self.state_size:] # returns a pair next state difference, action

    def next_state_action(self, s):
        """Returns a pair (next state, action)."""
        y = self.take_step(s)
        return y[:self.state_size] + s, y[self.state_size:] # returns a pair next state, action

    def next_action(self, s):
        """Returns the action."""
        y = self.take_step(s)
        return y[-self.action_size]

