import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

PI = np.pi

class AcrobotEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'acrobot.xml', 1, np.array([np.pi, 0]), np.array([0, 0]))
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward = -10*(ob[0]**2 + ob[1]**2) - 1*(ob[2]**2 + ob[3]**2) #- 0.0001*action[0]**2

        notdone = np.isfinite(ob).all() #and #and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def _get_obs(self):

        return np.concatenate([self.angle_normalize(self.sim.data.qpos), self.sim.data.qvel]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos, #+ self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
    
    def angle_normalize(self,x):
        return -((-x+PI) % (2*PI)) + PI