# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:04:07 2022

@author: Jean-Baptiste Bouvier

Main code to study inverted pendulum dynamics.

This code is inspired by the IEEE paper:
"Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"
by Anusha Nagabandi, Gregory Kahn, Ronald S. Fearing and Sergey Levine from UC Berkeley.
Available at https://ieeexplore.ieee.org/abstract/document/8463189

Overview:
First, we learn the dynamics of the system:
    given a state s(t) and an action a(t),
    the neural net predicts the next state difference s(t+dt)-s(t).
Second, we use Model Predictive Control (MPC) to design expert stabilizing trajectories.
Third, we use these expert trajectories to learn a stabilizing policy:
    given a state s(t), the neural net predicts the action to take a(t).
Fourth, we combine the policy and dynamics into a single neural network:
    given a state s(t), it predicts the action to take a(t) and
    the resulting next state difference s(t+dt)-s(t).
Fifth, we compare the trajectories resulting from the different dynamics and policies.
Sixth, we verify whether this combined system verifies Lipschitz and Lyapunov properties.
    NOT IMPLEMENTED yet.
"""


import torch
import pickle
import warnings
import numpy as np


from dynamics_learning import Dynamics_NN, Main_Dataset, Secondary_Dataset, training, relative_error, gain_Lip_Lyap_params
from expert_MPC import Expert_Dataset, MPC_query
from policy_learning  import Policy_Dataset, Policy_NN, policy_training, on_policy_rollout
from predictors import combined_NN
from trajectories_comparison import true_trajectory, traj_open_loop, traj_closed_loop, dynamics_MPC_traj, dynamics_policy_traj, predictor_traj, plot_trajectories#, plot_state_differences

# from ReLuVal.NN_Lip_Lyap_verif import LipLyap_maps, state2state_dynamics, state2state_ctrl_pend
# from ReLuVal.NN_Lip_Lyap_verif import is_NN_Lipschitz, is_NN_Lyapunov, sym_VS_brute_bounds

warnings.filterwarnings("ignore", message='Lazy modules are a new feature under heavy development ')





### Inverted pendulum system
class Pendulum():
    """Class for the inverted pendulum with its true parameters and true dynamics."""
    def __init__(self, dt):
        ### Parameters specific to the pendulum
        self.g = 9.81 # [m/s^2] gravity
        
        ### Parameters shared by all systems
        self.dt = dt # [s] time step for the discrete dynamics
        self.state_size = 2
        self.action_size = 1
        self.input_min = -torch.ones((1, self.action_size))
        self.input_max = torch.ones((1, self.action_size))
        self.state_labels = ['theta', 'theta dot']
        self.state_unit_labels = ['rad', 'rad/s']
        self.input_labels = ['torque']
        self.input_unit_labels = ['N']

    # continuous dynamics
    def derivative(self, s, a):
        """Calculates derivative of the pendulum's state s with action a."""
        if (a < self.input_min.squeeze()).max() or (a > self.input_max.squeeze()).max():
            print('The action is out of the admissible bounds')
        
        theta_dot = s[1]
        theta_ddot = 3*self.g*torch.sin(s[0])/2 + 45*a
        return torch.tensor([theta_dot, theta_ddot], dtype=torch.float)


    # discrete dynamics 
    def discrete_dynamics(self, s, a):
        """Calculates the next state difference with the true dynamics based on 
        current state s and action a. The output is s(t+dt) - s(t)."""
        return self.derivative(s, a)*self.dt
    
    # calculate next state
    def next_state(self, s, a):
        """Calculates the next state with the true dynamics based on 
        current state s and action a. The output is s(t+dt)."""
        return self.discrete_dynamics(s, a) + s
    
    # linearized dynamics
    def linearized_dynamics(self):
        """Calculates the A and B matrix of the linearized dynamics"""
        A = torch.tensor([[0., 1.], [3*self.g/2, 0.]])
        B = torch.tensor([[0.], [45.]])
        return A, B





#%% Learning the dynamics of the inverted pendulum
print("\nLearning pendulum dynamics\n")

### Global parameters
dt = 0.005 # size of the time step
N_train = 2**17 # 2**19 # number of data points for training
N_test = round(N_train/4) # number of data points for testing
N_pred = round(10./dt) # number of data points on the trajectory prediction
batch_size = 2**8 # number of samples studied before updating weights
num_hiddens = 2**6 # number of weights per hidden layer
epochs = 200 # number of training epochs
lr = 1e-2 # initial learning rate

pendulum = Pendulum(dt) # System to study
### Calculating the linear gain, Lipschitz constant and Lyapunov function of the real dynamics
[K, L, P] = gain_Lip_Lyap_params(pendulum)
### Creating neural network to model the dynamics of the pendulum
dynamics = Dynamics_NN(pendulum, num_hiddens)


################## Training network from scratch ##################
### Generating datasets
training_data = Main_Dataset(pendulum, N_train, random_actions = False)
testing_data = Secondary_Dataset(pendulum, N_test, training_data, random_actions = False)

### Saving the input and output means and stds of the training data
data_stats = {"input_mean": training_data.input_mean, "input_std": training_data.input_std, "output_mean": training_data.output_mean, "output_std": training_data.output_std}
fw = open('training_data_file', 'wb')
pickle.dump(data_stats, fw)
fw.close()
### Adding the statistics of the training data to the dynamics model for normalization
dynamics.update_data(data_stats) 

### Training of the dynamics model
training_loss, testing_loss = training(training_data, testing_data, dynamics, epochs, lr, batch_size)

relative_error(dynamics, training_data, 'training data')
relative_error(dynamics, testing_data, 'testing data')
### Saving the trained dynamics model
torch.save(dynamics.state_dict(), 'pendulum_dynamics.pt')


################# Reusing trained network ##################
# dynamics.load_state_dict(torch.load('pendulum_dynamics.pt'))
# dynamics.eval()
# data_stats = pickle.load( open('training_data_file', 'rb') )
# dynamics.update_data(data_stats)


num_params = dynamics.num_params()
print(f"Model has a total of {num_params:0} parameters, ratio of {N_train/num_params:.1f} datapoints to parameters.")






#%% Generating expert stable trajectories with Model Predictive Control (MPC)
### At each time step, uses random shooting and reward evalutation to select
### the best stable trajectories.
print("\nGenerating stable trajectories with MPC\n")

N_rep = 20 # number of times each action is repeated in the random shooting
H = 2 # horizon length for random shooting MPC
N_traj_shooting = 100 # number of trajectories sampled in random shooting
N_traj_aggreg = 2**7 # number of trajectories to compute during on-policy data aggregation
N_pred = 2**8 # length of the trajectory prediction

expert_dataset = Expert_Dataset(pendulum, dynamics, N_pred, N_traj_aggreg, N_traj_shooting, H, N_rep, P)
print(f"The on-policy data represents {100*expert_dataset.N/training_data.N:.2f}% of the training data")

### Adding the expert data to the initial training dataset
training_data.add_data(expert_dataset.X, expert_dataset.y)
### Retraining the neural network with this added data
training_loss, testing_loss = training(training_data, testing_data, dynamics, 50, lr, batch_size)




#%% Learning a stabilizing policy
### The neural network policy predicts the action to take at a given a state.
### This policy is trained on the expert MPC data.
print("\nLearning a stabilizing policy\n")

### Generating the training dataset and policy model
policy_data = Policy_Dataset(pendulum, expert_dataset)
policy = Policy_NN(pendulum, num_hiddens=64)

### Training the policy
training_loss = policy_training(policy_data, policy, epochs=200, lr=1e-3, batch_size = 2**6)
relative_error(policy, policy_data, 'policy data')
print(f"Model has a total of {policy.num_params():0} parameters, ratio of {policy_data.N/policy.num_params():.1f} datapoints to parameters.")

### Performing on-policy rollouts
N_traj = 2**3 # number of rollouts
N_pred = 2**7 # number of steps per rollouts
states = on_policy_rollout(pendulum, dynamics, policy, N_pred, N_traj)

### Querying the expert MPC to get the best actions to take at the visited states
inputs, labels = MPC_query(pendulum, dynamics, states, N_traj_shooting, H, N_rep, P)
policy_data.add_data(inputs, labels)
### Policy retraining
training_loss = policy_training(policy_data, policy, epochs=100, lr=1e-4, batch_size = round(policy_data.N/2**4))


### Adding datapoints close to the origin to improve convergence
states = 0.01*torch.randn((2**9, pendulum.state_size))
inputs, labels = MPC_query(pendulum, dynamics, states, N_traj_shooting, H, round(N_rep/2), P)
policy_data.add_data(inputs, labels)
### Policy retraining
training_loss = policy_training(policy_data, policy, epochs=100, lr=1e-4, batch_size = round(policy_data.N/2**4))




#%% Combining policy and dynamics into one single neural network
print("\nCombining policy and dynamics\n")
    
combined = combined_NN(dynamics, policy)

# ### Verification that the combined_NN and policy_NN + dynamics_NN are equivalent
# s = 0.2*torch.randn((1, pendulum.state_size)) # normal distribution of initial states
# a = policy.action(s)
# ds = dynamics.next_state_dif(s, a)
# comb_ds, comb_a = combined.next_state_dif_action(s)
# print(a.squeeze(), comb_a.squeeze(), ds.squeeze(), comb_ds)




#%% Comparison of the trajectories obtained with the different dynamic models and different policies
print("\nComparing trajectories\n")

### Parameters
N_pred = 2000 # number of prediction states, i.e., trajectory length
x0 = torch.tensor([0.1, 0.1]) # initial state
times = np.arange(N_pred)*dt 


### Trajectories storage
trajectories = [] # list to store the different trajectories
traj_labels = [] # list to store their labels
controls = [] # list to store the control inputs on each trajectories

### u=-Kx and true dynamics
true_traj, true_controls = true_trajectory(pendulum, x0, K, N_pred)
trajectories.append(true_traj)
controls.append(true_controls)
traj_labels.append('true')

### true_controls and NN_dynamics
open_loop_traj = traj_open_loop(x0, true_controls, dynamics, N_pred)
trajectories.append(open_loop_traj)
controls.append(true_controls)
traj_labels.append('open')

### u=-Kx and NN_dynamics
closed_loop_traj, closed_loop_controls = traj_closed_loop(x0, K, dynamics, N_pred)
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


### Plot all trajectories
plot_trajectories(times, trajectories, traj_labels, pendulum.state_labels, pendulum.state_unit_labels)
### Plot the difference in states between trajectories and the true one
# plot_state_differences(times, 'true', trajectories, traj_labels, pendulum.state_labels)
### Plot all controls
plot_trajectories(times[:-1], controls, traj_labels, pendulum.input_labels, pendulum.input_unit_labels)


















#%% Lyapunov and Lipschitz verifications of the combined_NN
# print("\nVerifying Lipschitz and Lyapunov properties\n")




# gamma = 0.01

# ### Verification of the Lyapunov and Lipschitz bounds, for the true trajectory
# [Lipschitz_respected, Lyapunov_respected] = Lip_Lyap_verification(true_traj, true_controls, P, gamma, L, N_pred)


# ### Verification whether the Neural Network satisfies the same Lipschitz and Lyapunov 
# ### conditions as the real dynamics, at least locally

# # Grid of the input set
# theta_step = 0.002
# theta_dot_step = 0.002

# theta_min = -0.3
# theta_max = 0.3
# theta_dot_min = -0.3
# theta_dot_max = 0.3

# x1_range = torch.arange(theta_min, theta_max, theta_step)
# x2_range = torch.arange(theta_dot_min, theta_dot_max, theta_dot_step)


# ### In-depth verification of the Lipschitz and Lyapunov conditions
# combined_s2s = state2state_ctrl_pend(combined) # creates a copy of the network but the output is just a state to apply symbolic intervals
# (Lipschitz_ratio, Lyapunov_ratio, Lipschitz_map, Lyapunov_map) = LipLyap_maps(combined_s2s.net, L, P, gamma, dt, x1_range, x2_range)


















# ### In-depth verification of the Lipschitz and Lyapunov conditions
# dynamics_s2s = state2state_dynamics(dynamics, K)
# (Lipschitz_ratio, Lyapunov_ratio, Lipschitz_map, Lyapunov_map) = LipLyap_maps(dynamics_s2s.net, L, P, gamma, dt, x1_range, x2_range)


# ## Fast verification of the Lipschitz and Lyapunov conditions
# # is_NN_Lyap = is_NN_Lyapunov(dynamics_s2s.net, P, gamma, dt, x1_range, x2_range)
# # is_NN_Lip = is_NN_Lipschitz(dynamics_s2s.net, L, dt, x1_range, x2_range)

 


# # ######### Adding bad data points to the training dataset ###############
# # # add_LipLyap_data(training_data, Lipschitz_map, L, Lyapunov_map, gamma, x1_range, x2_range, K, dt)

# # # ### Retraining the neural network with this added data
# # # training_loss, testing_loss = network_training(training_data, testing_data, dynamics, 50, lr, batch_size)

# # # torch.save(dynamics.state_dict(), 'model_NN_no_traj.pt')
# # # relative_error(training_data, 'training data')
# # # relative_error(testing_data, 'testing data')

# # # ### In-depth verification of the Lipschitz and Lyapunov conditions
# # # dynamics_s2s = state2state_dynamics(dynamics, K)
# # # (Lipschitz_ratio, Lyapunov_ratio, Lipschitz_map, Lyapunov_map) = LipLyap_maps(dynamics_s2s.net, L, P, gamma, dt, x1_range, x2_range)




# # ### Plot trajectory and sets where Lip and Lyap are not respected
# # traj_Lip_Lyap(traj, closed_loop_traj, norm_bound, Lipschitz_map, L, Lyapunov_map, gamma, x1_range, x2_range)














