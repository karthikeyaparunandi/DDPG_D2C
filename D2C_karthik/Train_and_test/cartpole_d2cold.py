#!/usr/bin/env python

import numpy as np
import gym
import time 
from D2C import D2C_class

ENV_NAME= 'Cartpole-v2'

env = gym.make(ENV_NAME)

horizon = 30
dt = 0.1
state_dimemsion = 4
control_dimension = 1
learning_rate = 0.0008
lr_0 = learning_rate
lr_schedule_k = 1
cost_gradient_conv_threshold_percent = 5
control_conv_threshold_percent = .8
sigma_del_u = 0.0005
J_nominal_prev = 0
Q = 0*np.eye(state_dimemsion)
Q_terminal = 1200*np.array([[2,0,0,0],[0,3,0,0],[0,0,.2,0],[0,0,0,.3]])
R = .0001*np.ones((control_dimension,control_dimension))
goal_state = np.zeros((state_dimemsion,))

init_time = time.time()

D2C = D2C_class(state_dimemsion, control_dimension, env, horizon, lr_0, lr_schedule_k, ctrl_state_freq_ratio=1)
U_nominal, X_nominal, J_nominal = D2C.D2C_run(  control_conv_threshold_percent, cost_gradient_conv_threshold_percent, \
	sigma_del_u, horizon, control_dimension, Q, Q_terminal, R, goal_state)

end_time = time.time()
time_taken = end_time - init_time
D2C.test_episode(U_nominal, horizon)
print(U_nominal, X_nominal, J_nominal, time_taken)

