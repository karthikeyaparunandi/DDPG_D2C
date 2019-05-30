# DDPG_D2C
Project to evaluate our D2C approach and compare it with DDPG

Contributers: Karthikeya S Parunandi and Ran Wang.

'karthik_branch' has DDPG [1] setup (Python3) (adapted from Keras-rl[2]'s implementation) and 'ran_branch' has the implementation of D2C (C++). Further, 'karthik_branch' also has the implementation of D2C in Python3. The following systems are considered as of now:
- Pendulum
- Cartpole
- Swimmer (3-link)
- Swimmer (6-link)
- Fish
- Hopper
- Cheetah

The models are taken from OpenAI gym [3] and Deepmind-Control suite[4] and then modified according to our problem.

References:
1) Continuous control with deep reinforcement learning, https://arxiv.org/abs/1509.02971
2) Keras-rl, https://github.com/keras-rl/keras-rl
3) OpenAI gym, https://github.com/openai/gym
4) Deepmind dm_control, https://github.com/deepmind/dm_control

