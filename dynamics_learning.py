# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:20:22 2023

@author: Jean-Baptiste Bouvier

Functions to learn the dynamics of a system 'sys' 
by training on a dataset of randomly generated triplets ( s(t), a(t), s(t+dt)-s(t) ).
"""

import torch
import scipy
import control
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F



# Generates random training data with the discrete dynamics of the system
def rand_data(sys, N, random_actions):
    """Generates training data with N random states and actions as inputs.
    The labels are the corresponding next state difference."""
    a_sz = sys.action_size
    s_sz = sys.state_size
    inputs = torch.zeros((N, a_sz + s_sz))
    labels = torch.zeros((N, s_sz))
    A, B = sys.linearized_dynamics()
    
    for i in range(N):
        
        s = torch.randn((1, s_sz)) # normal distribution of initial states
        if random_actions:
            a = (sys.input_max - sys.input_min)*torch.rand((1, a_sz)) + sys.input_min
        else:           
            if i%100 == 0: # changing gain K every 100 datapoints
                # random LQR weights to vary K
                LQR_Q = torch.diag(torch.rand(s_sz))
                LQR_R = torch.diag(torch.rand(a_sz)) #+ 10*LQR_Q.max() 
                K, _, _ = control.lqr(A, B, LQR_Q, LQR_R)
                K = torch.tensor(K, dtype=torch.float)
            a = torch.clamp(-s @ K.t(), min=sys.input_min, max=sys.input_max)
        
        inputs[i] = torch.cat((s, a), dim=1)
        labels[i] = sys.discrete_dynamics(s.squeeze(), a.squeeze())
    return [inputs, labels]



# Neural Network class to learn the dynamics of the system: (s(t), a(t)) -> s(t+dt)-s(t)
class Dynamics_NN(nn.Module):
    def __init__(self, sys, num_hiddens = 64):
        super().__init__()
        self.num_inputs = sys.state_size + sys.action_size
        self.num_outputs = sys.state_size
        self.state_size = sys.state_size
        self.action_size = sys.action_size
        
        self.num_hiddens = num_hiddens
        self.net = nn.Sequential(nn.Linear(self.num_inputs, self.num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(self.num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(self.num_hiddens), nn.ReLU(),
                                 nn.LazyLinear(self.num_outputs))
        
        # Statistical parameters of the training set to be updated
        self.input_mean = 0.
        self.input_std = 0.
        self.output_mean = 0.
        self.output_std = 0.
    
    # updates the statistics of the training set for normalization
    def update_data(self, data_stats):
        self.input_mean = data_stats['input_mean']
        self.input_std = data_stats['input_std']
        self.output_mean = data_stats['output_mean']
        self.output_std = data_stats['output_std']
    
    # returns the number of parameters used in the neural net
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        return self.net(x)
    
    # Calculates the predicted next state difference
    def next_state_dif(self, s, a):
        """Given a state and action returns the next state difference predicted by the model.
        To do so, it concatenates the state action-pair, normalizes them,
        applies the NN, and denormalizes the pair."""
        if s.shape[0] == 1:
            x = torch.cat((s,a), dim=1)
        else:
            x = torch.cat((s,a), dim=0)
        x = (x - self.input_mean)/self.input_std
        with torch.no_grad():
            return self.net(x)*self.output_std + self.output_mean
    
    # Calculates the predicted next state
    def next_state(self, s, a):
        """Given a state and action returns the next state predicted by the model"""
        return s + self.next_state_dif(s, a)
        
        
    

# class for the main dataset consisting in standardized inputs X and outputs y
# usually training dataset
class Main_Dataset(Dataset):
    def __init__(self, sys, N, random_actions):
        self.N = N
        [X, y] = rand_data(sys, N, random_actions) # generating random trajectories
        # Standardizing data
        self.input_mean = X.mean(dim=0)
        self.input_std = X.std(dim=0)
        self.X = (X - self.input_mean)/self.input_std
        
        self.output_mean = y.mean(dim=0)
        self.output_std = y.std(dim=0)
        self.y = (y - self.output_mean)/self.output_std
        
        # Adding Gaussian noise for robustness
        # self.X += torch.normal(0, 0.01, size=self.X.shape )
        # self.y += torch.normal(0, 0.01, size=self.y.shape )
        
    def __len__(self):# number of training examples
        return self.N
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]
    
    def add_data(self, inputs, labels):
        """Add unnormalized provided data to the training set."""
        inputs = (inputs - self.input_mean)/self.input_std
        labels = (labels - self.output_mean)/self.output_std
        self.X = torch.cat((self.X, inputs), dim=0)
        self.y = torch.cat((self.y, labels), dim=0)
        self.N += inputs.shape[0]
        
        
    
# Class for the secondary dataset standardized with the mean and std values from the main dataset
# Usually testing dataset
class Secondary_Dataset(Dataset):
    def __init__(self, sys, N2, main_dataset, random_actions):   
        # Generating random trajectories
        self.N = N2
        [X, y] = rand_data(sys, N2, random_actions) 
        # Standardizing data with parameters from the main dataset
        self.X = (X - main_dataset.input_mean)/main_dataset.input_std
        self.y = (y - main_dataset.output_mean)/main_dataset.output_std
       
    def __len__(self):# number of training examples
        return self.N
    
    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx,:]


# Training loop
def train_loop(dataloader, model, optimizer):
    """Train the network model on the dataset."""
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = F.mse_loss(pred, y) # loss = nn.MSELoss(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


# Testing loop
def test_loop(dataloader, model):
    """Test the performance of the model."""
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += F.mse_loss(pred, y).item() # test_loss += nn.MSELoss(pred, y).item()

    return test_loss/len(dataloader)


# Network training and plotting of the training and testing loss
def training(training_data, testing_data, model, epochs=30, lr=0.001, batch_size = 2**6):
    """Trains the network on the training dataset and verifies its performance on the testing dataset"""
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr)
    
    training_loss = np.zeros(epochs)
    testing_loss = np.zeros(epochs)
    
    plot_window = 50
    writing_window = 10
    
    for t in range(epochs):
        training_loss[t] = train_loop(train_dataloader, model, optimizer)
        testing_loss[t] = test_loop(test_dataloader, model)

        optimizer = torch.optim.SGD(model.parameters(), lr/(1+t)**0.5, momentum=0.5) # squarely decreasing lr
        
        if t%writing_window == 0:
            print(f'epoch: {t:3}  training loss: {training_loss[t]:10.8f}')    
        
        if t > 0 and t%plot_window == 0:
            plt.title('Dynamics training: loss')
            plt.plot(np.arange(t-plot_window, t), training_loss[t-plot_window:t], 'b', label='training loss')
            plt.plot(np.arange(t-plot_window, t), testing_loss[t-plot_window:t], 'g', label='testing loss')
            plt.legend(loc="upper right")
            plt.xlabel('epochs')
            plt.show()
      
    plt.title('Dynamics training: loss')
    plt.plot(np.arange(epochs), training_loss, 'b', label='training loss')
    plt.plot(np.arange(epochs), testing_loss, 'g', label='testing loss')
    plt.legend(loc="upper right")
    plt.xlabel('epochs')
    plt.show()

    print(f"Training loss stabilizing around {training_loss[round(epochs/2):].mean():.8f}")
    return training_loss, testing_loss



# Evaluates the relative error of the neural networks predictions
def relative_error(model, data, data_name='data'):
    """Calculates average and maximal relative error between true and learned dynamics over all data"""
    with torch.no_grad():
        Y_hat = model(data.X)
    rel_error = torch.abs((Y_hat - data.y)/data.y)*100 # relative error in percentage
    print(f"\nAverage relative error over {data_name} is {rel_error.mean().item():.2f} % and max relative error is {rel_error.max().item():.2f} %")



# Calculates the linear gain, Lipschitz constant and Lyapunov matrix.
def gain_Lip_Lyap_params(sys):
    """Calculates characteristics of the linearized dynamics:
    matrix K for the linear control u = -Kx,
    scalar L > 0 the Lipschitz constant ||dx/dt|| <= L||x||,
    matrix P > 0 for the Lyapunov function V(x)= xPx."""
    
    A, B = sys.linearized_dynamics()
    
    LQR_Q = torch.diag(torch.rand(sys.state_size))
    # make LQR_R bigger than LQR_Q to ensure small controller K to prevent saturation
    LQR_R = torch.diag(10*torch.rand(sys.action_size)) #+ 10*LQR_Q.max() 
    K, _, _ = control.lqr(A, B, LQR_Q, LQR_R)
    K = torch.tensor(K, dtype=torch.float)
    
    A_tilde = A - B @ K
    P = torch.tensor( control.lyap(A_tilde.t(), torch.eye(sys.state_size)), dtype=torch.float ) # Lyapunov matrix
    
    # matrix 2-norm is the max singular values + 3% of safety margin
    L = 1.03*max( scipy.linalg.svdvals(A_tilde) )
    return [K, L, P]
