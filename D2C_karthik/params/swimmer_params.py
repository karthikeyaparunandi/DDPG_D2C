'''
Inverted pendulum parameters

'''
import numpy as np


state_dimemsion = 10
control_dimension = 2

# Cost parameters for nominal design
Q = 9*np.diag([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
Q_terminal = 900*np.diag([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
R = .005*np.diag([2, 2])

## Mujoco simulation parameters
# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.01
horizon = 800
nominal_init_stddev = 0.1

# Cost parameters for feedback design

W_x_LQR = 10*np.eye(state_dimemsion)
W_u_LQR = 2*np.eye(control_dimension)
W_x_LQR_f = 100*np.eye(state_dimemsion)


# D2C parameters
feedback_n_samples = 50