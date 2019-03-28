import numpy as np
import gym
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, BatchNormalization
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, TestLogger
from matplotlib import pyplot as plt

ENV_NAME = 'Pendulum-v2'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

print(env.observation_space.shape)
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + (2,)))
#actor.add(BatchNormalization())
'''
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))

'''
actor.add(Dense(400))
actor.add(Activation('relu'))
actor.add(Dense(300))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + (2,), name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
#x = BatchNormalization()(x)
#x = Dense(32)(x)
#x = Activation('relu')(x)
'''
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
'''
x = Dense(400)(x)
x = Activation('relu')(x)
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)

critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


def build_callbacks(env_name):
    checkpoint_weights_filename = 'results/Pendulum/ddpg_' + env_name + '_weights_{step}.h5f'
    log_filename = 'results/Pendulum/ddpg_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=5000)]

    return callbacks

def build_test_callbacks(env_name):
    #checkpoint_weights_filename = 'ddpg_' + env_name + 'test_weights_{step}.h5f'
    #log_filename = 'ddpg_{}_test_log.json'.format(env_name)
    #callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=200)]
    callbacks = []#TestLogger()]
    #print(time.time())
    #print("heheheh")

    return callbacks

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step
n_samples = 1000
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=300000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.2)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=1.0, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mse'])

callbacks = build_callbacks(ENV_NAME)
test_callbacks = build_test_callbacks(ENV_NAME)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
#agent.fit(env, nb_steps=300000, visualize=False, callbacks=callbacks, verbose=1, nb_max_episode_steps=30)

# After training is done, we save the final weights.
#agent.save_weights('results/Pendulum/exp_4/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
agent.load_weights('results/Pendulum/exp_4/ddpg_{}_weights.h5f'.format(ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.
history, state_history_nominal, episode_reward = agent.test(env, nb_episodes=1, visualize=True, action_repetition=1, callbacks=test_callbacks, nb_max_episode_steps=30, std_dev_noise=0, initial_state=[np.pi, 0.])
#print(episode_reward, state_history_nominal)
u_max = 7.932
u_max = 2.98

'''
f = open("results/Pendulum/exp_2/data.txt", "a")

for i in frange(0.00, 1, 0.01):
    episode_reward_n = 0
    for j in range(n_samples):
        history, state_history, episode_reward = agent.test(env, nb_episodes=1, visualize=False, action_repetition=1, nb_max_episode_steps=10, initial_state=[np.pi, 0], std_dev_noise=i*u_max)
        episode_reward_n += episode_reward

    episode_reward_n_avg = episode_reward_n/n_samples
    f.write(str(i)+",\t"+str(np.linalg.norm(np.linalg.norm(state_history- state_history_nominal, axis=1)))+",\t"+str(episode_reward)+"\n")
    print(np.linalg.norm(np.linalg.norm(state_history- state_history_nominal, axis=1)), episode_reward)
f.close()


'''
'''
i, x, y = np.loadtxt('results/Pendulum/exp_2/data.txt', dtype=np.float64, delimiter=',\t',  unpack=True, usecols=(0,1,2))
i = 100*i
plt.figure(1)
plt.plot(i,x)
plt.xlabel("% Of max. signal (Standard deviation of perturbed noise)")
plt.ylabel("Trajectory mean squared error (Avergaed over {} samples)".format(n_samples))
plt.figure(2)
plt.xlabel("% Of max. signal (Standard deviation of perturbed noise)")
plt.ylabel("Episodic reward (Avergaed over {} samples)".format(n_samples))
plt.plot(i,y)
plt.show()
'''
