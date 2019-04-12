'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
python class for D2C method.
'''
#!/usr/bin/env python
import numpy as np
#import math

class D2C_class():

	def __init__(self, state_dimension, control_dimension, env):

		self.n_x = state_dimension
		self.n_u = control_dimension
		self.env = env



	def D2C_run(self, control_conv_threshold_percent, cost_gradient_conv_threshold_percent, sigma_del_u, horizon, control_dimension, learning_rate, Q, Q_terminal, R, goal_state):

		U_nominal_initial, X_nominal_initial, J_nominal_initial = self.initialize(horizon, Q, Q_terminal, R, control_dimension, goal_state)
		U_nominal, X_nominal, J_nominal = self.gradient_descent(control_conv_threshold_percent, cost_gradient_conv_threshold_percent, sigma_del_u, U_nominal_initial, \
								control_dimension, J_nominal_initial, horizon, Q, Q_terminal, R, learning_rate, goal_state)

		return U_nominal, X_nominal, J_nominal



	def initialize(self, horizon, Q, Q_terminal, R, control_dimension, goal_state):
		# initialization of the trajectory and cost
		U_nominal_initial = np.zeros((horizon, control_dimension))	#zero initialization of controls
		X_nominal_initial = self.forward_pass_sim(U_nominal_initial, horizon)
		J_nominal_initial = self.episodic_cost(X_nominal_initial, U_nominal_initial, Q, Q_terminal, R, horizon, goal_state)

		return U_nominal_initial, X_nominal_initial, J_nominal_initial



	def gradient_descent(self, control_conv_threshold_percent, cost_gradient_conv_threshold_percent, sigma_del_u, U_nominal_initial, \
					control_dim, J_nominal_initial, horizon, Q, Q_terminal, R, learning_rate, goal_state):
		# performs the gradient descent to find the optimal nominal
		dJ_nominal_percent = 100	
		J_nominal = J_nominal_initial
		U_nominal = U_nominal_initial
		dU_nominal = np.zeros(np.shape(U_nominal))
		dJ_nominal_percents = []
		dJ_nominal_percent_avg = dJ_nominal_percent
		i = 1
		gamma = 0.85


		while (dJ_nominal_percent_avg > control_conv_threshold_percent):
			X_nominal_extra = self.forward_pass_sim(U_nominal-gamma*learning_rate*dU_nominal, horizon)	

			J_nominal_extra = self.episodic_cost(X_nominal_extra, U_nominal-gamma*learning_rate*dU_nominal, Q, Q_terminal, R, horizon, goal_state)
			dU_nominal = (gamma)* dU_nominal + (1 - gamma)*(self.cost_gradient_calc(cost_gradient_conv_threshold_percent, \
												sigma_del_u, U_nominal-gamma*learning_rate*dU_nominal, control_dim, J_nominal_extra, Q, Q_terminal, R, horizon, goal_state))
			U_nominal += -learning_rate*dU_nominal	

			X_nominal = self.forward_pass_sim(U_nominal, horizon)	
			J_nominal_prev = J_nominal
			J_nominal = self.episodic_cost(X_nominal, U_nominal, Q, Q_terminal, R, horizon, goal_state)
			



			dJ_nominal = abs(J_nominal - J_nominal_prev)			
			dJ_nominal_percent = (dJ_nominal/J_nominal_prev)*100
			dJ_nominal_percents.append(dJ_nominal_percent)

			if(i%5 == 0):
				dJ_nominal_percent_avg = np.mean(dJ_nominal_percents)
				dJ_nominal_percents = []
			
			i += 1
			#print(J_nominal)
			#print(dJ_nominal_percent_avg, dJ_nominal_percents, i,"hiiiiiiiiiiiiiiii")

		return U_nominal, X_nominal, J_nominal
		



	def cost_gradient_calc(self, cost_gradient_conv_threshold_percent, sigma_del_u, U_nominal_prev, control_dim, J_nominal_prev, Q, Q_terminal, R, horizon, goal_state):
		# estimates the gradient of the cost function
		j = 0
		del_J = np.zeros(np.shape(U_nominal_prev))
		del_J_diff_percent = 100
		#alpha = 1.0
		control_traj_j_shape = np.shape(U_nominal_prev)

		while (del_J_diff_percent > cost_gradient_conv_threshold_percent):
		
			control_traj_j = U_nominal_prev + sigma_del_u*np.random.normal(0.0, 1.0, control_traj_j_shape)
			state_traj_j = 	self.forward_pass_sim(control_traj_j, horizon)
			del_J_next = (1 - (1/(j+1))) * del_J + (1/((j + 1)*(sigma_del_u**2)))*(self.episodic_cost(state_traj_j, control_traj_j, Q, \
															Q_terminal, R, horizon, goal_state) - J_nominal_prev)*(control_traj_j -  U_nominal_prev)
			
			if(np.linalg.norm(del_J) == 0):
				del_J_diff_percent = 100

			else:
				del_J_diff_percent = (np.max((del_J - del_J_next)/del_J))*100
			#print(del_J_diff_percent)
			del_J = del_J_next
			j += 1

		return del_J



	def episodic_cost(self, trajectory_states, trajectory_controls, Q, Q_terminal, R, horizon, goal_state):
		# calculates the episodic cost given the trajectory variables
		episodic_cost = 0
		
		for t in range(0, horizon):
			episodic_cost += self.cost(trajectory_states[t], trajectory_controls[t], Q, R, goal_state)

		episodic_cost += self.cost_terminal(trajectory_states[horizon], Q_terminal, R, goal_state)

		return episodic_cost



	def cost(self, state, control, Q, R, goal_state):
		# define the cost as a function of states and controls
		return (((state - goal_state) @ Q) @ (state - goal_state)) + ((control @ R) @ control)
		



	def cost_terminal(self, state, Q_terminal, R, goal_state):
		# define the terminal cost as a function of states and controls
		return (((state - goal_state) @ Q_terminal) @ (state - goal_state))
			


	def forward_pass_sim(self, U, horizon):
		# does a forward pass simulation with mujoco to find the states
		
		X = np.zeros((horizon+1, self.n_x))
		X[0] = self.env.reset()
		for i in range(0, horizon):
			a,b,c,d = self.env.step(U[i])
			X[i+1] = a
			
		return X



	def test_episode(self, controls, horizon):
		self.env.reset()
		for i in range(0, horizon):
			print(self.env.step(controls[i])[0])
			self.env.render(mode='human')


