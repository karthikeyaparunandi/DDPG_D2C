
import numpy as np
import scipy.linalg.blas as blas
from mujoco_py import load_model_from_path, MjSim, MjViewer


class ltv_sys_id_class():

	def __init__(self, model_xml_string,  state_size, action_size, n_samples=50):

		self.n_x = state_size
		self.n_u = action_size
		#perturbation sigma
		self.sigma = 0.00001
		model = load_model_from_path(model_xml_string)
		self.sim = MjSim(model, nsubsteps=1)
		#self.viewer = MjViewer(self.sim)
		#self.env = env
		self.n_samples = n_samples

	def sys_id(self, x_t, u_t):

		XU = np.random.normal(0.0, self.sigma, (self.n_samples, self.n_x + self.n_u))

		Y = np.transpose(self.simulate(np.transpose(x_t.reshape(-1, 1)) + XU[:, 0:self.n_x], np.transpose(u_t.reshape(-1, 1)) + XU[:, self.n_x:]) - \
						self.simulate(np.transpose(x_t.reshape(-1, 1)), np.transpose(u_t.reshape(-1, 1))))
		
		return (Y @ XU)@ np.linalg.inv(np.transpose(XU) @ XU)#(n_samples*self.sigma**2)#blas.dgemm(alpha=1.0, a=Y, b=X)


	def simulate(self, X, U):
		
		'''
		X - vector of states vertically stacked
		U - vector of controls vertically stacked
		'''
		assert X.shape[0] == U.shape[0]
		X_next = []
		old_state = self.sim.get_state()
		self.sim.reset()
		
		for i in range(X.shape[0]):

			self.sim.set_state_from_flattened(np.concatenate((np.array([old_state.time]), np.array(X[i]))))
			self.sim.forward()
			self.sim.data.ctrl[:] = U[i]
			self.sim.step()
			old_state = self.sim.get_state()		
			X_next.append(np.concatenate([self.angle_normalize(old_state.qpos), old_state.qvel]).ravel())
			
		return np.asarray(X_next)
	
	def traj_sys_id(self, x_nominal, u_nominal):	

		#assert x_nominal.shape[0] == u_nominal.shape[0]
		Traj_jac = []
		
		for i in range(u_nominal.shape[0]):
			
			Traj_jac.append(self.sys_id(x_nominal[i], u_nominal[i]))

		return np.asarray(Traj_jac)
		

	def angle_normalize(self,x):
		return -((-x+np.pi) % (2*np.pi)) + np.pi
