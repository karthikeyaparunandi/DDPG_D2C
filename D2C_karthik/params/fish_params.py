'''
Inverted pendulum parameters

'''
import numpy as np


state_dimemsion = 27
control_dimension = 5

# Cost parameters for nominal design
Q = 9*np.diag(np.concatenate([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.zeros((13,))]))
Q_terminal = 1500*np.diag(np.concatenate([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.zeros((13,))]))
R = .05*np.diag([2, 2, 2, 2, 2])

## Mujoco simulation parameters
# Number of substeps in simulation
ctrl_state_freq_ratio = 5
dt = 0.01
horizon = 600
nominal_init_stddev = 0.1

# Cost parameters for feedback design

W_x_LQR = 10*np.eye(state_dimemsion)
W_u_LQR = 2*np.eye(control_dimension)
W_x_LQR_f = 100*np.eye(state_dimemsion)


# D2C parameters
feedback_n_samples = 50