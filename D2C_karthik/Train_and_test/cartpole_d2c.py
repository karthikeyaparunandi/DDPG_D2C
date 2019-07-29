#!/usr/bin/env python

import numpy as np
import gym
from D2C import D2C_class
import time
from ltv_sys_id import ltv_sys_id_class
from params.cartpole_params import *
import os

class cartpole_D2C(D2C_class, ltv_sys_id_class):
	
	def __init__(self, state_dimension, control_dimension, MODEL_XML, horizon, lr_0, lr_schedule_k, ctrl_state_freq_ratio=ctrl_state_freq_ratio):
		'''
			Declare the matrices associated with the cost function
		'''
		self.X_p_0 = np.zeros((state_dimension,))

		D2C_class.__init__(self, state_dimension, control_dimension, MODEL_XML, horizon, lr_0, lr_schedule_k, ctrl_state_freq_ratio=ctrl_state_freq_ratio)
		ltv_sys_id_class.__init__(self, MODEL_XML, state_dimemsion, control_dimension, n_samples=feedback_n_samples)


	def state_output(self, state):
		'''
			Given a state in terms of Mujoco's MjSimState format, extract the state as a numpy array, after some preprocessing
		'''
		return np.concatenate([self.angle_normalize(state.qpos), state.qvel])

	def angle_normalize(self, x):
		'''
		Function to normalize the cartpole's angle from [0, Inf] to [-np.pi, np.pi]
		'''
		return -((-x+np.pi) % (2*np.pi)) + np.pi


	def initialize(self, horizon, Q, Q_terminal, R, control_dimension, goal_state):

		# initialization of the trajectory and cost

		U_nominal_initial = np.random.normal(0, nominal_init_stddev, (horizon, control_dimension))#np.zeros((horizon, control_dimension))	#zero initialization of controls
		X_nominal_initial = self.forward_pass_sim(U_nominal_initial, horizon)
		J_nominal_initial = self.episodic_cost(X_nominal_initial, U_nominal_initial, Q, Q_terminal, R, horizon, goal_state)

		return U_nominal_initial, X_nominal_initial, J_nominal_initial




if __name__=="__main__":

	path_to_D2C = "/home/karthikeya/Documents/research/DDPG_D2C/D2C_karthik"
	MODEL_XML = path_to_D2C + "/models/cartpole.xml"
	path_to_policy = path_to_D2C + "/experiments/cartpole/exp_1/cartpole_D2C_policy.txt"
	training_cost_data_file = path_to_D2C + "/experiments/cartpole/exp_1/cartpole_D2C_training_cost_history.txt"
	path_to_data = path_to_D2C + "/experiments/cartpole/exp_1/cartpole_D2C_data.txt"
	path_to_exp = path_to_D2C + "/experiments/cartpole/exp_1/"

	with open(path_to_data, 'a') as f:

		f.write("D2C training performed for an inverted cartpole task:\n\n")

		f.write("System details : {}\n".format(os.uname().sysname + "--" + os.uname().nodename + "--" + os.uname().release + "--" + os.uname().version + "--" + os.uname().machine))
		f.write("-------------------------------------------------------------\n")

	# learning rate
	lr_0 =  0.0008
	lr_schedule_k = 1
	cost_gradient_conv_threshold_percent = 1
	control_conv_threshold_percent = .02
	
	# Standard deviation of the control perturbation \del u
	sigma_del_u = 0.0005
	J_nominal_prev = 0

	initial_state = np.array([0, np.pi-0.1, 0, 0])
	goal_state = np.zeros((state_dimemsion, ))
	
	init_time = time.time()
	total_time = 0


	for i in range(1):

		# Declare the D2C class
		D2C = cartpole_D2C(state_dimemsion, control_dimension, MODEL_XML, horizon, lr_0, lr_schedule_k, ctrl_state_freq_ratio=ctrl_state_freq_ratio)
		
		time_1 = time.time()

		# Run the D2C algorithm for nominal trajectory
		U_nominal, X_nominal, J_nominal = D2C.D2C_run(control_conv_threshold_percent, cost_gradient_conv_threshold_percent, \
														sigma_del_u, horizon, control_dimension, Q, Q_terminal, R, initial_state, goal_state)
		
		time_2 = time.time()
		
		# System identification and linear feedback design
		AB = D2C.traj_sys_id(X_nominal, U_nominal)
		K = D2C.feedback(AB, W_x_LQR, W_u_LQR, W_x_LQR_f)

		time_3 = time.time()

		open_loop_time = time_2 - time_1
		feedback_time = time_3 - time_2
		D2C_algorithm_run_time = open_loop_time + feedback_time

		# Save the episodic cost
		with open(training_cost_data_file, 'w') as f:

			for cost in D2C.episodic_cost_history:

				f.write("%s\n" % cost)

		D2C.test_episode(U_nominal, X_nominal, K, horizon)
		D2C.save_policy(U_nominal, X_nominal, K, path_to_policy)
		
		total_time += D2C_algorithm_run_time
		print(X_nominal)
		
		with open(path_to_data, 'a') as f:

			f.write("Time to compute open-loop: {},\nTime to compute feedback:{},\nTotal time taken: {}\n".format(open_loop_time, feedback_time, D2C_algorithm_run_time))
			f.write("------------------------------------------------------------------------------------------------------------------------------------\n")

		# Testing phase of the resulting trajectory
		D2C.plot_episodic_cost_history(save_to_path=path_to_exp+"/episodic_cost_OL_training.png")
		
		print("total time taken:",total_time)


