import numpy as np
import gym
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, BatchNormalization, Lambda
from keras.optimizers import Adam
from keras import initializers
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, TestLogger
from gym.wrappers.monitoring.video_recorder import VideoRecorder

ENV_NAME = 'HalfCheetah-v2'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
#np.random.seed(123)
#env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# actor model.
state_input = Input(shape=(1,) + env.observation_space.shape, name='state_input')
flattened_state = Flatten()(state_input)
y = Dense(400)(flattened_state)
y = Activation('relu')(y)
y = Dense(300)(y)
y = Activation('relu')(y)
y = Dense(nb_actions, kernel_initializer=initializers.RandomUniform(minval=-0.003,  maxval=0.003))(y)
y = Activation('tanh')(y)
y = Lambda(lambda x: x )(y)
actor = Model(inputs=[state_input], outputs=y)
print(actor.summary())

# critic model
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(400)(x)
x = Activation('relu')(x)
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1, kernel_initializer=initializers.RandomUniform(minval=0.0003,   maxval=0.0003))(x)
x = Activation('linear')(x)

critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


def build_callbacks(env_name):
    checkpoint_weights_filename = '../results/HalfCheetah/exp_1/ddpg_' + env_name + '_weights_{step}.h5f'
    log_filename = '../results/HalfCheetah/exp_1/ddpg_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=5000)]

    return callbacks

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

GAMMA = 0.999               # GAMMA of our cumulative reward function
STEPS_PER_EPISODE = 900     # No. of time-steps per episode

# configure and compile our agent by using built-in Keras optimizers and the metrics!
# allocate the memory by specifying the maximum no. of samples to store
memory = SequentialMemory(limit=1500000, window_length=1)
# random process for exploration noise
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, dt=0.01, mu=0., sigma=.25, sigma_min=0.05, n_steps_annealing=1000000)

print(env.action_space.high)
# define the DDPG agent
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=GAMMA, target_model_update=1e-3)
# compile the model
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mse'])

callbacks = build_callbacks(ENV_NAME)

# ----------------------------------------------------------------------------------------------------------------------------------------
# Training phase

# fitting the agent
#agent.fit(env, nb_steps=1500000, visualize=False, callbacks=callbacks, verbose=1, gamma=GAMMA, nb_max_episode_steps=STEPS_PER_EPISODE)

# After training is done, we save the final weights.
#agent.save_weights('../results/HalfCheetah/exp_1/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Testing phase

agent.load_weights('../results/HalfCheetah/exp_1/ddpg_HalfCheetah-v2_weights_1500000.h5f'.format(ENV_NAME))

# Finally, evaluate our algorithm.
history, state_history_nominal, episode_reward_nominal = agent.test(env, nb_episodes=1, visualize=True, action_repetition=1, nb_max_episode_steps=STEPS_PER_EPISODE, \
                                                         initial_state=np.zeros((17,)), std_dev_noise=0, gamma=GAMMA)
print(episode_reward_nominal, state_history_nominal)

# -----------------------------------------------------------------------------------------------------------------------------------------