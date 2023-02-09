# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:57:04 2023

@author: Jean-Baptiste Bouvier

Computing and comparing trajectories generated differently:
true dynamics, learned dynamics, linear controller, MPC controller, learned controller,...
"""


import torch
from matplotlib import pyplot as plt

from expert_MPC import best_action





### Using u=-Kx and true dynamics to propagate trajectory
def true_trajectory(sys, x0, K, N_pred):
    """Generates a trajectory of length 'N_pred' starting from x0
    propagated through u=-Kx and the true dynamics.""" 
    traj = torch.zeros((N_pred, sys.state_size))
    controls = torch.zeros((N_pred-1, sys.action_size))
    traj[0] = x0
    
    for i in range(N_pred-1):
        x = traj[i]
        u = -K @ x
        controls[i] = u
        traj[i+1] = sys.next_state(x, u)
        
    return traj, controls


### Using true_controls and NN_dynamics to propagate trajectory
def traj_open_loop(x0, true_controls, dynamics, N_pred):
    """Generates a trajectory of length 'N_pred' starting from x0
    propagated through true_controls and dynamics neural networks."""
    traj = torch.zeros((N_pred, dynamics.num_outputs))
    traj[0] = x0
    
    for i in range(N_pred-1):
        traj[i+1] = dynamics.next_state(traj[i], true_controls[i])
            
    return traj


### Using u=-Kx and NN_dynamics to propagate trajectory
def traj_closed_loop(x0, K, dynamics, N_pred):
    """Generates a trajectory of length 'N_pred' starting from x0
    propagated through u=-Kx and dynamics neural networks."""
    traj = torch.zeros((N_pred, dynamics.state_size))
    actions = torch.zeros((N_pred-1, dynamics.action_size))
    traj[0] = x0
    
    for i in range(N_pred-1):
        s = traj[i]
        actions[i] = -K @ s.t()
        traj[i+1] = dynamics.next_state(s, actions[i])

    return traj, actions


### Using MPC controller and NN_dynamics to propagate trajectory
def dynamics_MPC_traj(sys, x0, dynamics, N_pred, N_traj_shooting, H, N_rep, M):
    """Generates a trajectory of length 'N_pred' starting from x0
    propagated through the dynamics neural network and MPC shooting."""

    traj = torch.zeros((N_pred, dynamics.state_size))
    actions = torch.zeros((N_pred-1, dynamics.action_size))
    traj[0] = x0

    for i in range(N_pred-1):
        actions[i] = best_action(sys, dynamics, traj[i].unsqueeze(dim=0), N_traj_shooting, H, N_rep, M)
        traj[i+1] = dynamics.next_state(traj[i], actions[i])
        
    return traj, actions 


### Using NN_policy and NN_dynamics to propagate trajectory
def dynamics_policy_traj(dynamics, policy, x0, N_pred):
    """Generates a trajectory of length 'N_pred' starting from x0
    propagated through the policy and dynamics neural networks."""
   
    traj = torch.zeros((N_pred, dynamics.state_size))
    actions = torch.zeros((N_pred-1, dynamics.action_size))
    traj[0] = x0
   
    for i in range(N_pred-1):
        actions[i] = policy.action(traj[i])
        traj[i+1] = dynamics.next_state(traj[i], actions[i])
    
    return traj, actions


### Using a combined predictor NN to propagate trajectory
def predictor_traj(predictor, x0, N_pred):
    """Generates a trajectory of length 'N_pred' starting from x0
    propagated through a combined predictor neural network calculating both
    the next state and action given a current state."""
   
    traj = torch.zeros((N_pred, predictor.state_size))
    actions = torch.zeros((N_pred-1, predictor.action_size))
    traj[0] = x0
   
    for i in range(N_pred-1):
        traj[i+1], actions[i] = predictor.next_state_action(traj[i])
    
    return traj, actions



### Plot all trajectories together for comparison
def plot_trajectories(times, trajectories, traj_labels, fig_labels, unit_labels):
    """Plots all given trajectories with their associated labels.
    trajectories is a list of traj
    traj_labels is a list of text labels associated to each traj
    fig_labels is a list of text labels associated to each state of the traj, i.e., each figure."""
    
    # Same number of trajectories and labels
    assert(len(trajectories) == len(traj_labels) )
    num_figures = trajectories[0].shape[1]
    num_traj = len(trajectories)
    
    for fig_id in range(num_figures):
    
        plt.title(fig_labels[fig_id])
        for traj_id in range(num_traj):
            plt.plot(times, trajectories[traj_id][:,fig_id].detach().numpy(), label = traj_labels[traj_id])
        plt.xlabel('time (s)')
        plt.ylabel(fig_labels[fig_id] + ' (' + unit_labels[fig_id] + ')')
        plt.legend()
        plt.show()
    



### Plots the difference between the states of each trajectories and a reference one
def plot_state_differences(times, ref_name, trajectories, traj_labels, fig_labels):
    """Plots the difference between the states of each trajectories with respect to the reference one.
    The name of the reference trajectory is the input 'ref_name'.
    'trajectories' is a list of traj
    'traj_labels' is a list of text labels associated to each traj
    'fig_labels' is a list of text labels associated to each state of the traj, i.e., each figure."""
    
    # Same number of trajectories and labels
    assert(len(trajectories) == len(traj_labels) )
    num_traj = len(trajectories)
    num_figures = trajectories[0].shape[1]
    
    ### Determining the reference trajectory to compare with the others
    for traj_id in range(num_traj):
        if traj_labels[traj_id] == ref_name:
            ref_traj = trajectories[traj_id]
    
    ### Creating a figure for each state
    for fig_id in range(num_figures):
        
        plt.title(fig_labels[fig_id] + ' difference')
        for traj_id in range(num_traj):
            if traj_labels[traj_id] == ref_name:
                continue
            else:
                ### Calculating the distance between the reference trajectory and the current one
                state_dif = (ref_traj[:,fig_id] - trajectories[traj_id][:,fig_id]).detach().numpy()
                plt.plot(times, state_dif, label = traj_labels[traj_id])
    
        plt.xlabel('time (s)')
        plt.legend()
        plt.show()


