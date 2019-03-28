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
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
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
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
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

def time_array(darray, episodes_per_echo):
        tarray = []
        for d in range(0, len(darray)-1):
            ind = np.arange(darray[d], darray[d] + darray[d+1], (darray[d+1])/episodes_per_echo)
            tarray = [tarray, ind]

        return tarray

n_samples = 5000
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=200000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.25)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=1.0, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mse'])

callbacks = build_callbacks(ENV_NAME)
test_callbacks = build_test_callbacks(ENV_NAME)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
#agent.fit(env, nb_steps=200000, visualize=False, callbacks=callbacks, verbose=1, nb_max_episode_steps=30)

# After training is done, we save the final weights.
#agent.save_weights('results/Pendulum/exp_6/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
agent.load_weights('results/Pendulum/exp_6/ddpg_{}_weights.h5f'.format(ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.
history, state_history_nominal, episode_reward_nominal = agent.test(env, nb_episodes=1, visualize=True, action_repetition=1, callbacks=test_callbacks, nb_max_episode_steps=30, std_dev_noise=0, initial_state=[np.pi, 0.])
print(episode_reward_nominal, state_history_nominal)
u_max = 4.905

'''
f = open("results/Pendulum/exp_5/data.txt", "a")

for i in frange(0.00, 0.4, 0.01):
    episode_reward_n = 0
    Var_n = 0
    for j in range(n_samples):
        history, state_history, episode_reward = agent.test(env, nb_episodes=1, visualize=False, action_repetition=1, nb_max_episode_steps=30, initial_state=[np.pi, 0], std_dev_noise=i*u_max)
        episode_reward_n += episode_reward
        Var_n += (episode_reward)**2

    episode_reward_n_avg = episode_reward_n/n_samples
    var_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_n - episode_reward_n**2)

    if var_avg > 0 :
        std_dev_avg = np.sqrt(var_avg)
    else:
        print("hii")
        std_dev_avg = 0

    f.write(str(i)+",\t"+str(np.linalg.norm(np.linalg.norm(state_history- state_history_nominal, axis=1)))+",\t"+str(episode_reward_n_avg)+",\t"+str(std_dev_avg)+"\n")
    print(np.linalg.norm(np.linalg.norm(state_history- state_history_nominal, axis=1)), episode_reward_n_avg, std_dev_avg)
f.close()


'''
'''
i, x, y, z = np.loadtxt('results/Pendulum/exp_5/data.txt', dtype=np.float64, delimiter=',\t',  unpack=True, usecols=(0,1,2,3))
i = 100*i
plt.figure(1)
plt.plot(i,x)
plt.xlabel("% Of max. signal (Standard deviation of perturbed noise)")
plt.ylabel("Trajectory mean squared error (Avergaed over {} samples)".format(n_samples))
plt.figure(2)
plt.xlabel("% Of max. signal (Standard deviation of perturbed noise)", fontsize=16)
plt.ylabel("Episodic reward fraction (Avg. over {} samples)".format(n_samples), fontsize=16)
plt.plot(i,y/-460.50434061412034, 'bo-')
plt.plot(i, (z+y)/-460.50434061412034, 'r-')
plt.plot(i, (y-z)/-460.50434061412034, 'r-')
plt.fill_between(i, (z+y)/-460.50434061412034, (y-z)/-460.50434061412034, facecolor='yellow')
plt.grid()
plt.show()
'''
