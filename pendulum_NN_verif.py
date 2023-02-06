# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:04:07 2022

@author: bouvier3
"""

import scipy
import torch
import pickle
# from control import lqr, lyap
import warnings
import numpy as np
# from torch import nn
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle


### Import of homemade functions
from dynamics_learning import Dynamics_NN, Main_Dataset, Secondary_Dataset, training, relative_error
from RL_policy_training  import OnPolicy_Dataset, Policy_Data, Policy_NN, policy_training, on_policy_rollout, MPC_query
from trajectories_comparison import true_trajectory, NN_traj_open_loop, NN_traj_closed_loop, dynamics_MPC_traj, dynamics_policy_traj, predictor_traj, plot_trajectories
from predictors import Predictor_NN, Predictor_Dataset, combined_NN
from ReLuVal.NN_Lip_Lyap_verif import LipLyap_maps, state2state_dynamics, state2state_ctrl_pend
# from ReLuVal.NN_Lip_Lyap_verif import is_NN_Lipschitz, is_NN_Lyapunov, sym_VS_brute_bounds



warnings.filterwarnings("ignore", message='Lazy modules are a new feature under heavy development ')
### LEARNING Inverted pendulum dynamics
### No trajectories, only input pairs






class Pendulum():
    """Class for the inverted pendulum with its true parameters and true dynamics."""
    def __init__(self, dt):
        ### Parameters specific to the quadcopter
        self.g = 9.81 # [m/s^2] gravity
        
        ### Parameters common to all systems
        self.dt = dt # [s] time step for the discrete dynamics
        self.state_size = 2
        self.action_size = 1
        self.input_min = -torch.ones((1, self.action_size))
        self.input_max = torch.ones((1, self.action_size))
        self.state_labels = ['theta', 'theta dot']
        self.input_labels = ['u']
        self.squared_inputs = False # needed for the linearization

    # continuous dynamics
    def derivative(self, s, a):
        """Calculates derivative of the pendulum's state s with action a."""
        if (a < self.input_min.squeeze()).max() or (a > self.input_max.squeeze()).max():
            print('The action is out of the admissible bounds')
        
        theta_dot = s[1]
        theta_ddot = 3*self.g*torch.sin(s[0])/2 + 3*15*a
        return torch.tensor([theta_dot, theta_ddot], dtype=torch.float)


    # discrete dynamics 
    def discrete_dynamics(self, s, a):
        """Calculates the next state difference with the true dynamics based on 
        current state s and action a. The output is s(t+dt) - s(t)."""
        return self.derivative(s, a)*self.dt
    
    # 
    def next_state(self, s, a):
        """Calculates the next state with the true dynamics based on 
        current state s and action a. The output is s(t+dt)."""
        return self.discrete_dynamics(s, a) + s
    
    #
    def linearized_dynamics(self):
        """Calculates the A and B matrix of the linearized dynamics"""
        A = torch.tensor([[0, 1], [3*self.g/2, 0]])
        B = torch.tensor([[0], [15*3]])
        return A, B



# Transient parameters for the linearized inverted pendulum dynamics
def transient_bd_params(sys, LQR_Q = torch.eye(2), LQR_R = torch.eye(1)):
    """Calculates parameters needed for the transient bounds with linearized dynamics, i.e.,
    matrix K for the linear control u = -Kx,
    matrix P>0 for the Lyapunov function V(x)=x'Px,
    gamma>0 ensuring exponential decrease of V: dV/dt <= -2 gamma V,
    L>0 the Lipschitz constant ||f(x)|| <= L||x||"""
    
    A, B = sys.linearized_dynamics()
    
    ### K gain matrix
    K, _, _ = lqr(A, B, LQR_Q, LQR_R)
    K = torch.tensor(K, dtype=torch.float)
    
    A_tilde = A-B*K
    
    Q = torch.eye(sys.state_size)
    P = torch.tensor( lyap(A_tilde.t(), Q), dtype=torch.float ) # Lyapunov matrix
    gamma = min(np.linalg.eigvals(Q))/( 2 * max(np.linalg.eigvals(P))) # exponential coefficient for the Lyapunov differential inequality
    
    # matrix 2-norm is the max singular values + 3% of safety margin for the sin(theta) linearization
    L = 1.03*max( scipy.linalg.svdvals(A_tilde) )
    return [K, P, gamma, L]





### Global parameters
dt = 0.005 # size of the time step
N_train = 2**17 # 2**19 # number of data points for training
N_test = round(N_train/4) # number of data points for testing
N_pred = round(10./dt) # number of data points on the trajectory prediction
batch_size = 2**8 # number of samples studied before updating weights
num_hiddens = 2**6 # number of weights per hidden layer
epochs = 100 # 300 # number of training epochs
lr = 1e-2 # initial learning rate

pendulum = Pendulum(dt)
# Create neural network to model the dynamics of the pendulum
dynamics = Dynamics_NN(pendulum, num_hiddens)


################## Training network from scratch ##################
### Generating datasets
training_data = Main_Dataset(pendulum, N_train, random_actions = False)
testing_data = Secondary_Dataset(pendulum, N_test, training_data, random_actions = False)

### Saving the input and output means and stds 
data_stats = {"input_mean": training_data.input_mean, "input_std": training_data.input_std, "output_mean": training_data.output_mean, "output_std": training_data.output_std}
fw = open('training_data_file', 'wb')
pickle.dump(data_stats, fw)
fw.close()
dynamics.update_data(data_stats) # add the data stats to the model

### Training
training_loss, testing_loss = training(training_data, testing_data, dynamics, epochs, lr, batch_size)
print(f"Training loss stabilizing around {training_loss[round(epochs/2):].mean():.8f}")
torch.save(dynamics.state_dict(), 'pendulum_dynamics.pt')
relative_error(dynamics, training_data, 'training data')
relative_error(dynamics, testing_data, 'testing data')


################# Reusing trained network, no dataset needed ##################
# dynamics.load_state_dict(torch.load('pendulum_dynamics.pt'))
# dynamics.eval()
# data_stats = pickle.load( open('training_data_file', 'rb') )
# dynamics.update_data(data_stats)


num_params = dynamics.num_params()
print(f"Model has a total of {num_params:0} parameters, ratio of {N_train/num_params:.1f} datapoints to parameters.")



### Random LQR weights to generate different controllers K
LQR_Q = torch.diag_embed(torch.rand(1,2)).squeeze()
LQR_R = 10*torch.rand(1,1) + 10*LQR_Q.max() # make LQR_R bigger than LQR_Q to ensure small controller K to prevent saturation
[K, P, gamma, L] = transient_bd_params(pendulum, LQR_Q, LQR_R)





#%% RL training



N_rep = 20 # number of times each action is repeated in the random shooting
H = 2 # horizon length for random shooting MPC
N_traj_shooting = 100 # number of trajectories sampled in random shooting
N_traj_aggreg = 2**1 #2**8 # number of trajectories to compute during on-policy data aggregation
N_pred = 2**8 # length of the trajectory prediction

on_policy_dataset = OnPolicy_Dataset(pendulum, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, P)
print(f"The on-policy data represents {100*on_policy_dataset.N/training_data.N:.2f}% of the training data")


# ### Adding the on-policy aggregated data to the initial training dataset
# training_data.add_data(on_policy_dataset.X, on_policy_dataset.y)
# ### Retraining the neural network with this added data
# training_loss, testing_loss = network_training(training_data, testing_data, dynamics, 50, lr, batch_size)




#%% Imitation Learning
### This phase aims at building a policy as a neural network predicting the
### action to take given a step. This policy is based on the 'expert'
### trajectories obtained at the RL phase.


### This policy takes a state as input and returns an action to take as output.
policy_data = Policy_Data(pendulum, on_policy_dataset)
policy = Policy_NN(pendulum, num_hiddens=32)

# Training the policy
training_loss = policy_training(policy_data, policy, epochs=100, lr=0.005, batch_size = 2**6)
relative_error(policy, policy_data, 'policy data')
print(f"Model has a total of {policy.num_params():0} parameters, ratio of {policy_data.N/policy.num_params():.1f} datapoints to parameters.")

# Performing on-policy rollouts, i.e., trajectory propagation
N_traj = 2**1 # 2**3 # number of trajectories to propagate, i.e., number of rollouts
N_pred = 2**7 # number of steps per trajectories (rollouts)
states = on_policy_rollout(pendulum, dynamics, policy, N_pred, N_traj)


# Querying the “expert” MPC controller for “true” action labels for visited states
inputs, labels = MPC_query(pendulum, dynamics, states, N_traj_shooting, H, N_rep, P)
policy_data.add_data(inputs, labels)
# Policy retraining
training_loss = policy_training(policy_data, policy, epochs=30, lr=0.005, batch_size = round(policy_data.N/2**2))


# Add datapoints close to the origin to improve convergence
states = 0.01*torch.randn((2**9, pendulum.state_size))
inputs, labels = MPC_query(pendulum, dynamics, states, N_traj_shooting, H, N_rep, P)
policy_data.add_data(inputs, labels)
# Policy retraining
training_loss = policy_training(policy_data, policy, epochs=30, lr=0.005, batch_size = round(policy_data.N/2**2))




#%% Instead of model-free learning, just combining the NN policy and dynamics into one NN

    
combined = combined_NN(dynamics, policy)

### Verification that the combined_NN and policy_NN + dynamics_NN are equivalent
s = 0.2*torch.randn((1, pendulum.state_size)) # normal distribution of initial states
a = policy.action(s)
ds = dynamics.next_state_dif(s, a)
comb_ds, comb_a = combined.next_state_dif_action(s)
print(a, comb_a, ds, comb_ds)



#%% Model-free learning
### This phase consists in training a model-free neural network, which given a
### state can predict the next step and action.
### This network is built on the policy obtained at the previous phase and on 
### the system's dynamics obtained at the initial phase.




N_ctrl = 2**16 # number of data points to train the controlled pendulum on
N_test_ctrl = round(N_ctrl/4)
batch_size = 2**8
epochs = 50
lr = 0.001
controlled_pendulum = Predictor_NN(pendulum, num_hiddens=64)
controlled_data = Predictor_Dataset(policy, dynamics, N_ctrl)
testing_ctrl_data = Predictor_Dataset(policy, dynamics, N_test_ctrl)
training_loss, testing_loss = training(controlled_data, testing_ctrl_data, controlled_pendulum, epochs, lr, batch_size)

relative_error(controlled_pendulum, controlled_data, 'controlled pendulum training data')
relative_error(controlled_pendulum, testing_ctrl_data, 'controlled pendulum testing data')

num_params = controlled_pendulum.num_params()
print(f"The controlled pendulum NN has a total of {num_params:0} parameters, ratio of {N_ctrl/num_params:.1f} datapoints to parameters.")



#%% All trajectories comparisons
### N_pred steps per trajectories
### N_pred-1 steps per control sequences
###
### true_trajectory: true dynamics + u = -Kx
### traj_dyn_cl: NN_dynamics + u = -Kx
### traj_dyn_ol: NN_dynamics + true controls applied exactly
###
###
###


### Plots the distance between trajectories and the maximal allowed value
def plot_traj_distances(times, norm_bound, ol_traj_distance, cl_traj_distance):
    
    ol = torch.Tensor.numpy(ol_traj_distance)
    # Plot the open loop distance only until it passes the bound to keep a legible graph
    diverge_index = (np.nonzero( ol[2:] > max(norm_bound) ))[0][0]
    
    plt.title('Trajectory distances')
    plt.plot(times, norm_bound, label = "norm bound" )
    plt.plot(times[:diverge_index], ol[:diverge_index], label = "open loop")
    plt.plot(times, cl_traj_distance.detach().numpy(), label = "closed loop")
    plt.xlabel('time (s)')
    plt.legend()
    plt.show()


# Transient bounds calculation
def transient_bound(P, gamma, L, x0, times):
    """Calculation of the norm bound between trajectories"""
    L1 = ((1 - np.exp(-gamma*times)) * L/gamma)
    L2 = np.exp(-gamma*times)
    min_bound = np.vstack( (L1, L2) ).min(axis=0)
    return 2*np.sqrt( np.dot(x0, P@x0) ) * min_bound / np.sqrt(min(np.linalg.eigvals(P)))






################### Parameters #####################
N_pred = 2000
x0 = torch.tensor([0.1, 0.1])
times = np.arange(N_pred)*dt


# ############ Trajectories computations #############
trajectories = [] # list to store the different trajectories
traj_labels = [] # list to store their labels
controls = [] # list to store the control inputs on each trajectories

### u=-Kx and true dynamics
true_traj, true_controls = true_trajectory(pendulum, x0, K, N_pred)
trajectories.append(true_traj)
controls.append(true_controls)
traj_labels.append('true')

### true_controls and NN_dynamics
open_loop_traj = NN_traj_open_loop(x0, true_controls, dynamics, N_pred)
trajectories.append(open_loop_traj)
controls.append(true_controls)
traj_labels.append('open')

### u=-Kx and NN_dynamics
closed_loop_traj, closed_loop_controls = NN_traj_closed_loop(x0, K, dynamics, N_pred)
trajectories.append(closed_loop_traj)
controls.append(closed_loop_controls)
traj_labels.append('closed')

### MPC controls and NN_dynamics
mpc_traj, mpc_controls = dynamics_MPC_traj(pendulum, x0, dynamics, N_pred, N_traj_shooting, H, N_rep, P)
trajectories.append(mpc_traj)
controls.append(mpc_controls)
traj_labels.append('mpc')

### policy controls and NN_dynamics
policy_traj, policy_controls = dynamics_policy_traj(dynamics, policy, x0, N_pred)
trajectories.append(policy_traj)
controls.append(policy_controls)
traj_labels.append('policy')

### combined policy and dynamics
comb_traj, comb_controls = predictor_traj(combined, x0, N_pred)
trajectories.append(comb_traj)
controls.append(comb_controls)
traj_labels.append('combined')

### NN_controlled_pendulum
ctrl_traj, ctrl_controls = predictor_traj(controlled_pendulum, x0, N_pred)
trajectories.append(ctrl_traj)
controls.append(ctrl_controls)
traj_labels.append('ctrl')


# ############ Trajectories comparisons ###############

### Plot all trajectories
plot_trajectories(times, trajectories, traj_labels, pendulum.state_labels)
### Plot all controls
plot_trajectories(times[:-1], controls, traj_labels, pendulum.input_labels)





### Transient bound
norm_bound = transient_bound(P, gamma, L, x0, times)
### Distance between the true trajectory and the learned one
ol_traj_distance = torch.norm(true_traj - open_loop_traj, dim=1)
cl_traj_distance = torch.norm(true_traj - closed_loop_traj, dim=1)
### Plot of the distances allowed by the theory and actual distance between trajectories
plot_traj_distances(times, norm_bound, ol_traj_distance, cl_traj_distance)

















#%% Lyapunov and Lipschitz verifications


# Lyapunov and Lipschitz bounds are calculated with the linearized dynamics, while we propagate the nonlinear dynamics
# Verification of the Lyapunov and Lipschitz bounds, i.e., gamma and L
def Lip_Lyap_verification(sys, traj, controls, P, gamma, L, N_pred):
    """Verifies whether the Lipschitz and Lyapunov bounds hold during the whole trajectory.
    Indeed, these bounds are calculated with the linearized dynamics, while the trajectory is propagated with the true nonlinear dynamics."""
    Lipschitz_respect = np.zeros((N_pred-1,1))
    Lyapunov_respect = np.zeros((N_pred-1,1))
    
    for i in range(N_pred-1):
        x = traj[i]
        u = controls[i]    
        dx_dt = sys.derivative(x,u) # continuous nonlinear derivative
        Lipschitz_respect[i] = int( torch.norm(dx_dt) <= L * torch.norm(x)) 
        Lyapunov_respect[i] = int( 2 * torch.dot(x, P @ dx_dt ) <= -2 * gamma * torch.dot(x, P @ x))
   
    # plt.scatter(dt*np.arange(N_pred-1), Lipschitz_respect+0.01, s=10, c='b', label="Lipschitz respect")
    # plt.scatter(dt*np.arange(N_pred-1), Lyapunov_respect-0.01, s=10, c='r', label="Lyapunov respect")
    # plt.xlabel('time (s)')
    # plt.ylim(-0.01, 1.01)
    # plt.legend()
    # plt.show()
    
    ### return booleans
    # return [bool(min(Lipschitz_respect)), bool(min(Lyapunov_respect))] 
    if bool(min(Lipschitz_respect)):
        print("The true trajectory verifies the Lipschitz condition.")
    else:
        print("The true trajectory does NOT verify the Lipschitz condition.")
    if bool(min(Lyapunov_respect)):
        print("The true trajectory verifies the Lyapunov condition.")
    else:
        print("The true trajectory does NOT verify the Lyapunov condition.")



### Plot trajectory and sets where Lip and Lyap are not respected
def traj_Lip_Lyap(traj, closed_loop_traj, norm_bound, Lipschitz_map, L, Lyapunov_map, gamma, x1_range, x2_range):
    """Plot the system trajectories in the phase portrait to see if it intersects
    with the sets where the Lipschitz and Lyapunov conditions are not respected."""
    
    width = x1_range[1] - x1_range[0]
    height = x2_range[1] - x2_range[0]
    
    fig, ax = plt.subplots()
    ax.plot()
    plt.title('Phase portrait')
    # Rectangle where the Lipschitz and Lyapunov conditions have been tested
    ax.add_patch(Rectangle( (x1_range[0], x2_range[0]), x1_range[-1]-x1_range[0], x2_range[-1]-x2_range[0], facecolor='w',  edgecolor = 'silver'))
    # Areas where the Lipschitz and Lyapunov conditions are not satisfied
    for i1 in range(len(x1_range)):
        for i2 in range(len(x2_range)):
            local_L = Lipschitz_map[i1, i2]
            local_gamma = Lyapunov_map[i1, i2]
            if local_L > L or local_gamma < gamma:
                ax.add_patch(Rectangle( (x1_range[i1], x2_range[i2]), width, height, color = 'r'))
    # Plots the reachable area around true trajectory according to the norm bound
    for i in range(len(norm_bound)):
        ax.add_patch( Circle(traj[i].detach().numpy(), radius=norm_bound[i], color='lavender') )
    # Plot true trajectory and learned closed loop trajectory
    plt.plot(traj[:,0].detach().numpy(), traj[:,1].detach().numpy(), color = 'b', label = "true")
    plt.plot(closed_loop_traj[:,0].detach().numpy(), closed_loop_traj[:,1].detach().numpy(), color = 'g', label = "closed loop")
    
    plt.xlabel('theta')
    plt.ylabel('theta dot')
    plt.legend()
    plt.show()



# Add to the training set the data points where the Lipschitz and Lyapunov bound do not hold
def add_LipLyap_data(sys, training_data, Lipschitz_map, L, Lyapunov_map, gamma, x1_range, x2_range, K):
    """Add to the training set the data points where either the Lipschitz or Lyapunov conditions do not hold,
    in the hope of improving the network performance at these points."""
    
    factor = 1.0 # if 1.1, select only data points where either Lip > L + 10% or Lyap < gamma-10%
    indices = torch.cat((torch.nonzero(Lipschitz_map > L*factor), torch.nonzero(Lyapunov_map < gamma/factor)))
    N = indices.shape[0]
    inputs = torch.zeros((N,3))
    labels = torch.zeros((N,2))

    for i in range(N):
        s = torch.tensor(( [[x1_range[indices[i,0]]], [x2_range[indices[i,1]]]]))
        u = - K @ s
            
        inputs[i,:] = torch.cat((s, u), dim=0).t()
        labels[i,:] = sys.discrete_dynamics(s.squeeze(), u)
        
    training_data.add_data(inputs, labels)













### Verification of the Lyapunov and Lipschitz bounds, for the true trajectory
[Lipschitz_respected, Lyapunov_respected] = Lip_Lyap_verification(true_traj, true_controls, P, gamma, L, N_pred)


### Verification whether the Neural Network satisfies the same Lipschitz and Lyapunov 
### conditions as the real dynamics, at least locally

# Grid of the input set
theta_step = 0.002
theta_dot_step = 0.002

theta_min = -0.3
theta_max = 0.3
theta_dot_min = -0.3
theta_dot_max = 0.3

x1_range = torch.arange(theta_min, theta_max, theta_step)
x2_range = torch.arange(theta_dot_min, theta_dot_max, theta_dot_step)

### In-depth verification of the Lipschitz and Lyapunov conditions
ctrl_pend_s2s = state2state_ctrl_pend(controlled_pendulum) # creates a copy of the network but the output is just a state to apply symbolic intervals
(Lipschitz_ratio, Lyapunov_ratio, Lipschitz_map, Lyapunov_map) = LipLyap_maps(ctrl_pend_s2s.net, L, P, gamma, dt, x1_range, x2_range)

### In-depth verification of the Lipschitz and Lyapunov conditions
combined_s2s = state2state_ctrl_pend(combined) # creates a copy of the network but the output is just a state to apply symbolic intervals
(Lipschitz_ratio, Lyapunov_ratio, Lipschitz_map, Lyapunov_map) = LipLyap_maps(combined_s2s.net, L, P, gamma, dt, x1_range, x2_range)


















### In-depth verification of the Lipschitz and Lyapunov conditions
dynamics_s2s = state2state_dynamics(dynamics, K)
(Lipschitz_ratio, Lyapunov_ratio, Lipschitz_map, Lyapunov_map) = LipLyap_maps(dynamics_s2s.net, L, P, gamma, dt, x1_range, x2_range)


## Fast verification of the Lipschitz and Lyapunov conditions
# is_NN_Lyap = is_NN_Lyapunov(dynamics_s2s.net, P, gamma, dt, x1_range, x2_range)
# is_NN_Lip = is_NN_Lipschitz(dynamics_s2s.net, L, dt, x1_range, x2_range)

 


# ######### Adding bad data points to the training dataset ###############
# # add_LipLyap_data(training_data, Lipschitz_map, L, Lyapunov_map, gamma, x1_range, x2_range, K, dt)

# # ### Retraining the neural network with this added data
# # training_loss, testing_loss = network_training(training_data, testing_data, dynamics, 50, lr, batch_size)

# # torch.save(dynamics.state_dict(), 'model_NN_no_traj.pt')
# # relative_error(training_data, 'training data')
# # relative_error(testing_data, 'testing data')

# # ### In-depth verification of the Lipschitz and Lyapunov conditions
# # dynamics_s2s = state2state_dynamics(dynamics, K)
# # (Lipschitz_ratio, Lyapunov_ratio, Lipschitz_map, Lyapunov_map) = LipLyap_maps(dynamics_s2s.net, L, P, gamma, dt, x1_range, x2_range)




# ### Plot trajectory and sets where Lip and Lyap are not respected
# traj_Lip_Lyap(traj, closed_loop_traj, norm_bound, Lipschitz_map, L, Lyapunov_map, gamma, x1_range, x2_range)














