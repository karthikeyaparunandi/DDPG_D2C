'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
python class for D2C method.
'''
#!/usr/bin/env python
import numpy as np
from  multiprocessing import Process, Queue, cpu_count
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimPool
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import json

#import math

class D2C_class():

	def __init__(self, state_dimension, control_dimension, model_xml_string, horizon, lr_0, lr_schedule_k, ctrl_state_freq_ratio=1):

		self.n_x = state_dimension
		self.n_u = control_dimension
		self.horizon = horizon
		self.n_processes = np.min([cpu_count(), horizon])
		self.lr_0 = lr_0
		self.lr_schedule_k = lr_schedule_k
		self.episodic_cost_history = []

		self.sim = MjSim(load_model_from_path(model_xml_string), nsubsteps=ctrl_state_freq_ratio)


	def D2C_run(self, control_conv_threshold_percent, cost_gradient_conv_threshold_percent, sigma_del_u, horizon, control_dimension, Q, Q_terminal, R, initial_state, goal_state):

		self.X_p_0 = initial_state

		U_nominal_initial, X_nominal_initial, J_nominal_initial = self.initialize(horizon, Q, Q_terminal, R, control_dimension, goal_state)
		U_nominal, X_nominal, J_nominal = self.gradient_descent(control_conv_threshold_percent, cost_gradient_conv_threshold_percent, sigma_del_u, U_nominal_initial, \
								control_dimension, J_nominal_initial, horizon, Q, Q_terminal, R, goal_state)

		return U_nominal, X_nominal, J_nominal



	def gradient_descent(self, control_conv_threshold_percent, cost_gradient_conv_threshold_percent, sigma_del_u, U_nominal_initial, \
					control_dim, J_nominal_initial, horizon, Q, Q_terminal, R, goal_state):
		
		# performs the gradient descent to find the optimal nominal

		dJ_nominal_percent = 100	
		J_nominal = J_nominal_initial
		U_nominal = U_nominal_initial
		dU_nominal = np.zeros(np.shape(U_nominal))
		dJ_nominal_percents = []
		dJ_nominal_percent_avg = dJ_nominal_percent
		i = 1
		gamma = 0.9
		learning_rate = self.lr_0
	
		try:
			while (dJ_nominal_percent_avg > control_conv_threshold_percent):

				X_nominal_extra = self.forward_pass_sim(U_nominal-gamma*learning_rate*dU_nominal, horizon)	

				J_nominal_extra = self.episodic_cost(X_nominal_extra, U_nominal-gamma*learning_rate*dU_nominal, Q, Q_terminal, R, horizon, goal_state)
				dU_nominal = (gamma)* dU_nominal + (1 - gamma)*(self.cost_gradient_calc(cost_gradient_conv_threshold_percent, \
													sigma_del_u, U_nominal - learning_rate*gamma*dU_nominal, control_dim, J_nominal_extra, Q, Q_terminal, R, horizon, goal_state))
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
				learning_rate = self.exponential_lr_schedule(self.lr_0, i, self.lr_schedule_k)
				self.episodic_cost_history.append(J_nominal)	

		except:

			print("U_nominal", U_nominal, "X_nominal", X_nominal, "J_nominal", J_nominal)

		return U_nominal, X_nominal, J_nominal
		


	def cost_gradient_calc_partial(self, queue, sigma_del_u, U_nominal_prev, control_dim, J_nominal_prev, Q, Q_terminal, R, horizon, goal_state, control_traj_j_shape):

		control_traj_j = U_nominal_prev + sigma_del_u*np.random.RandomState().normal(0.0, 1.0, control_traj_j_shape)
		state_traj_j = 	self.forward_pass_sim(control_traj_j, horizon)

		queue.put((self.episodic_cost(state_traj_j, control_traj_j, Q, Q_terminal, R, horizon, goal_state) - J_nominal_prev)*(control_traj_j -  U_nominal_prev)	)



	def cost_gradient_calc(self, cost_gradient_conv_threshold_percent, sigma_del_u, U_nominal_prev, control_dim, J_nominal_prev, Q, Q_terminal, R, horizon, goal_state):
		# estimates the gradient of the cost function

		j = 0
		del_J = np.zeros(np.shape(U_nominal_prev))
		del_J_diff_percent = 100
		control_traj_j_shape = np.shape(U_nominal_prev)
		

		while (del_J_diff_percent > cost_gradient_conv_threshold_percent):
			
			del_u = sigma_del_u*np.random.normal(0.0, 1.0, control_traj_j_shape)
			#ck = c / (j + 1)**gamma
			#del_u = ck*(2*bernoulli.rvs(size=control_traj_j_shape, p=0.5)-1)
			control_traj_j_f = U_nominal_prev + del_u
			#control_traj_j_b = U_nominal_prev - del_u
			
			state_traj_j_f = 	self.forward_pass_sim(control_traj_j_f, horizon)
			#state_traj_j_b = 	self.forward_pass_sim(control_traj_j_b, horizon)

			# using central differnce formula to update the sample gradient
			#print((del_u @ del_u.T)/sigma_del_u**2)
			del_J_next = (1 - (1/(j+1))) * del_J + (1/((j + 1)*(sigma_del_u**2)*control_dim))*(self.episodic_cost(state_traj_j_f, control_traj_j_f, Q, \
															Q_terminal, R, horizon, goal_state) - J_nominal_prev)*(del_u)

			# del_J_next = (1 - (1/(j+1))) * del_J + (1/((j + 1) *control_traj_j_shape[1]))*(self.episodic_cost(state_traj_j_f, control_traj_j_f, Q, \
			# 												Q_terminal, R, horizon, goal_state) - self.episodic_cost(state_traj_j_b, control_traj_j_b, Q, \
			# 												Q_terminal, R, horizon, goal_state))/(2*del_u)
			# print(self.episodic_cost(state_traj_j_f, control_traj_j_f, Q, \
			# 												Q_terminal, R, horizon, goal_state) - self.episodic_cost(state_traj_j_b, control_traj_j_b, Q, \
			# 												Q_terminal, R, horizon, goal_state), del_u)
			# Processes = []
			# q = Queue()
			# for kth_process in range(self.n_processes):								
			# 	p = Process(target=self.cost_gradient_calc_partial, args=(q, sigma_del_u, U_nominal_prev, control_dim, J_nominal_prev, Q, Q_terminal, R, horizon, goal_state, control_traj_j_shape,))
			# 	Processes.append(p)
			# 	p.start()


			# for kth_process in Processes:
			# 	kth_process.join()
			# 	del_J_next = (1 - (1/(j+1))) * del_J + (1/((j + 1)*(sigma_del_u**2)))*q.get()
			# 	#print((self.episodic_cost(state_traj_j, control_traj_j, Q, Q_terminal, R, horizon, goal_state) - J_nominal_prev)*(control_traj_j -  U_nominal_prev)		, Processes[kth_process].get(),"this")
			# 	j += 1		
			# 	#print(self.cost_gradient_calc_partial(sigma_del_u, U_nominal_prev, control_dim, J_nominal_prev, Q, Q_terminal, R, horizon, goal_state, control_traj_j_shape,))			#Processes[kth_process].get())#del_J_next, del_J, "this")	
			# 	if(np.linalg.norm(del_J) == 0):
			# 		del_J_diff_percent = 100
					
			# 	else:
			# 		del_J_diff_percent = (np.max((del_J - del_J_next)/del_J))*100
									
			# 	#print(j, del_J_diff_percent)
				
			# 	del_J = del_J_next

			if(np.linalg.norm(del_J) == 0):

					del_J_diff_percent = 100

			else:

				del_J_diff_percent = (np.max((del_J - del_J_next)/del_J))*100
				
			
			del_J = del_J_next
			j += 1

		return del_J


	def episodic_cost_partial(self, trajectory_states, trajectory_controls, Q, R, horizon, goal_state, kth_process):

		for t in range(kth_process, horizon, self.n_processes):

				episodic_cost_partial += self.cost(trajectory_states[t], trajectory_controls[t], Q, R, goal_state)

		return episodic_cost_partial


	def episodic_cost(self, trajectory_states, trajectory_controls, Q, Q_terminal, R, horizon, goal_state, multiprocess_flag=0):
		# calculates the episodic cost given the trajectory variables
		if multiprocess_flag:
			episodic_cost_partial = np.zeros((self.n_processes, 1))
			episodic_cost = 0
			Processes = []

			with mp.Pool(processes=self.n_processes) as pool:
				for kth_process in range(self.n_processes):
					p = pool.apply_async(episodic_cost_partial, (trajectory_states, trajectory_controls, Q, R, horizon, goal_state, kth_process,))
					Processes.append(p)	

			for kth_process in range(self.n_processes):
				episodic_cost += Processes[kth_process].get()
			episodic_cost += self.cost_terminal(trajectory_states[horizon], Q_terminal, R, goal_state)

			return episodic_cost

		else:

			episodic_cost = sum(self.cost(trajectory_states[t], trajectory_controls[t], Q, R, goal_state) for t in range(0, horizon)) +\
							 		self.cost_terminal(trajectory_states[horizon], Q_terminal, R, goal_state)

			return episodic_cost



	def cost(self, state, control, Q, R, goal_state):
		# define the cost as a function of states and controls
		return (((state - goal_state) @ Q) @ (state - goal_state)) + ((control @ R) @ control)
		


	def cost_terminal(self, state, Q_terminal, R, goal_state):
		# define the terminal cost as a function of states and controls
		return (((state - goal_state) @ Q_terminal) @ (state - goal_state))
			



	def forward_pass_sim(self, U, horizon, x_nominal=None, feedback_gain=None, render=0):
		
		################## defining local functions & variables for faster access ################

		sim = self.sim
		
		##########################################################################################
		sim.set_state_from_flattened(np.concatenate([np.array([self.sim.get_state().time]), self.X_p_0.flatten()]))
		X = np.zeros((horizon+1, self.n_x))
		np.copyto(X[0], self.X_p_0)

		for t in range(0, horizon):
			
			sim.forward()
			if feedback_gain is None:

				sim.data.ctrl[:] = U[t].flatten()

			else:

				sim.data.ctrl[:] = U[t].flatten() + feedback_gain[t] @ (X[t] - x_nominal[t])

			sim.step()
			X[t+1] = self.state_output(sim.get_state())

			if render:

				sim.render(mode='window')

		return X

	
	def test_episode(self, u_nominal, x_nominal, feedback_gain, horizon, render=1, path=None):
		
		'''
			Test the episode using the current policy if no path is passed. If a path is mentioned, it simulates the controls from that path
		'''
		
		if path is None:
		
			self.forward_pass_sim(u_nominal, horizon, x_nominal, feedback_gain, render=render)
		
		else:
		
			self.sim.set_state_from_flattened(np.concatenate([np.array([self.sim.get_state().time]), self.X_p_0.flatten()]))
			
			with open(path) as f:

				Pi = json.load(f)

			for i in range(0, self.N):
				
				self.sim.forward()
				self.sim.data.ctrl[:] = np.array(Pi['U'][str(i)]).flatten() + np.array(Pi['K'][str(i)]) @ (self.state_output(self.sim.get_state()) - np.array(Pi['X'][str(i)]))
				self.sim.step()
				
				if render:
					self.sim.render(mode='window')


	def plot_episodic_cost_history(self, save_to_path=None):

		try:
			self.plot_(np.asarray(self.episodic_cost_history).flatten(), save_to_path=save_to_path)

		except:

			print("Plotting failed")
			pass

	def plot_(self, y, x=None, show=1, save_to_path=None):

		if x==None:

			# params = { 'axes.labelsize': 15, 'font.size': 20, 'legend.fontsize': 15, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'text.usetex': True ,\
			# 		    'figure.figsize': [7, 5], 'font.weight': 'bold', 'axes.labelweight': 'bold', 'ps.useafm' : True, 'pdf.use14corefonts':True, 'pdf.fonttype': 42, 'ps.fonttype': 42}

			plt.figure(figsize=(7, 5))
			plt.plot(y, linewidth=2)
			plt.xlabel('Training iteration count', fontweight="bold", fontsize=12)
			plt.ylabel('Episodic cost', fontweight="bold", fontsize=12)
			#plt.grid(linestyle='-.', linewidth=1)
			plt.grid(color='.910', linewidth=1.5)
			plt.title('Open-loop training : Episodic cost vs No. of training iterations')
			if save_to_path is not None:
				print(save_to_path)
				plt.savefig(save_to_path, format='png')#, dpi=1000)
			plt.tight_layout()
			plt.show()
		
		else:

			plt.plot(y, x)
			plt.show()


	def save_policy(self, u_nominal, x_nominal, feedback_gain, path_to_file):

		Pi = {}
		# Open-loop part of the policy
		Pi['U'] = {}
		# Closed loop part of the policy - linear feedback gains
		Pi['K'] = {}
		Pi['X'] = {}

		for t in range(0, self.horizon):
			
			Pi['U'][t] = np.ndarray.tolist(u_nominal[t])
			Pi['K'][t] = np.ndarray.tolist(feedback_gain[t])
			Pi['X'][t] = np.ndarray.tolist(x_nominal[t])
			
		with open(path_to_file, 'w') as outfile:  

			json.dump(Pi, outfile)


	def exponential_lr_schedule(self, alpha_0, t, k):

		return alpha_0*(1 + np.exp(-k*t))


	def feedback(self, AB, W_x_LQR, W_u_LQR, W_x_LQR_f):
		'''
		AB matrix comprises of A and B as [A | B] stacked at every ascending time-step
	
		'''

		P = W_x_LQR_f

		K = [np.zeros((self.n_u, self.n_x))]#np.zeros((self.n_u, self.n_x, self.horizon))

		for t in range(self.horizon-1, 0, -1):

			A = AB[t, :, 0:self.n_x]
			B = AB[t, :, self.n_x:]

			S = W_u_LQR + ( (np.transpose(B) @ P) @ B)

			# LQR gain 
			K.append(-np.linalg.inv(S) @ ( (np.transpose(B) @ P) @ A))
			
			# second order equation

			P = W_x_LQR  +  ((np.transpose(A) @ P) @ A) - ((np.transpose(K[-1]) @ S) @ K[-1]) 


		return np.asarray(K)

	def initialize(self, horizon, Q, Q_terminal, R, control_dimension, goal_state):
		# initialization of the trajectory and cost
		pass
