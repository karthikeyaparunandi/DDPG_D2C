'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
python class for D2C method.
'''
#!/usr/bin/env python
import numpy as np
import math

class D2C_class():

	def __init__(self, horizon, state_dimension, control_dimension, learning_rate, dt, env):

		self.N = horizon
		self.n_x = state_dimension
		self.n_u = control_dimension
		self.alpha = learning_rate
		self.dt = dt
		self.env = env

	def initialize(self, horizon):
		# initialization of the trajectory with zeros
		self.U_nominal = np.zeros((horizon, self.n_u))
		self.X_nominal = self.forward_pass_sim(self.U_nominal)	

	def gradient_descent(self,):
		# performs the gradient descent to find the optimal nominal
		self.U_nominal += -self.alpha*self.cost_gradient_calc()
		
		self.X_nominal = self.forward_pass_sim(self.U_nominal)	

	def cost_gradient_calc(self, cost_gradient_conv_threshold_percent, sigma_del_u, U_nominal_prev, control_dim, J_nominal_prev):
		# estimates the gradient of the cost function
		j = 0
		del_J = np.zeros(control_dim)
		del_J_diff_percent = 100

		while (del_J_diff_percent < cost_gradient_conv_threshold_percent):
			
			control_traj_j = U_nominal_prev + sigma_del_u*np.random.normal(0.0, 1.0, control_dim)
			state_traj_j = 	self.forward_pass_sim(control_traj_j)
			del_J_next = (1 - (1/(j+1))) * del_J + (1/((j + 1)*sigma_del_u))*(self.episodic_cost(state_traj_j, control_traj_j, Q, Q_terminal, R) - J_nominal_prev)*(control_traj_j - U_nominal_prev)
			del_J_diff_percent = (abs(del_J - del_J_next)/del_J)*100
			del_J = del_J_next
			j += 1
			

	def episodic_cost(self, trajectory_states, trajectory_controls, Q, Q_terminal, R):
		# calculates the episodic cost given the trajectory variables
		episodic_cost = 0
		for t in range(0, horizon):
			episodic_cost += self.cost(trajectory_states[t], trajectory_controls[t], Q, R)

		episodic_cost += self.cost_terminal(trajectory_states[horizon], Q_terminal, R)

		return episodic_cost

	def cost(self, state, control, Q, R):
		# define the cost as a function of states and controls
		return ((state @ Q) @ state) + ((control @ R) @ control)
		

	def cost_terminal(self, state, Q_terminal, R):
		# define the terminal cost as a function of states and controls
		return ((state @ Q_terminal) @ state) 
			
	def forward_pass_sim(self, U):
		# does a forward pass simulation with mujoco to find the states
		
		X = np.zeros((self.N, self.n_x))
		
		for i in range(0, horizon):
			X[i] = self.env.step(U[i])

		return X

