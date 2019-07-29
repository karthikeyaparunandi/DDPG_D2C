'''
Inverted pendulum parameters

'''
import numpy as np


state_dimemsion = 2
control_dimension = 1

# Cost parameters for nominal design
Q = 0*np.array([[1,0],[0,0.2]])
Q_terminal = 900*np.array([[1,0],[0,0.1]])
R = .01*np.ones((1,1))

## Mujoco simulation parameters
# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.1
horizon = 30
nominal_init_stddev = 0.01

# Cost parameters for feedback design

W_x_LQR = 10*np.eye(2)
W_u_LQR = 2*np.eye(1)
W_x_LQR_f = 100*np.eye(2)


# D2C parameters
feedback_n_samples = 20