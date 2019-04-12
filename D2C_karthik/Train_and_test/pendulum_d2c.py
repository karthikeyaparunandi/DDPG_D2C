#!/usr/bin/env python

import numpy as np
import gym
from D2C import D2C_class

ENV_NAME= 'Pendulum-v2'

env = gym.make(ENV_NAME)

horizon = 30
dt = 0.1
state_dimemsion = 2
control_dimension = 1
learning_rate = 0.0008
cost_gradient_conv_threshold_percent = 5
control_conv_threshold_percent = .8
sigma_del_u = 0.0005
J_nominal_prev = 0
Q = 0*np.eye(2)
Q_terminal = 700*np.eye(2)
R = .0001*np.ones((1,1))
goal_state = np.zeros((2,))

D2C = D2C_class(state_dimemsion, control_dimension, env)
U_nominal, X_nominal, J_nominal = D2C.D2C_run(  control_conv_threshold_percent, cost_gradient_conv_threshold_percent, \
	sigma_del_u, horizon, control_dimension, learning_rate, Q, Q_terminal, R, goal_state)
D2C.test_episode(U_nominal, horizon)
print(U_nominal, X_nominal, J_nominal)

