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
from matplotlib import pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder

ENV_NAME = 'Swimmer-v2'

action_bound = 12
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
#np.random.seed(123)
#env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
state_input = Input(shape=(1,) + env.observation_space.shape, name='state_input')
flattened_state = Flatten()(state_input)
#y = BatchNormalization()(flattened_state)
y = Dense(400)(flattened_state)
y = Activation('relu')(y)
#y = BatchNormalization()(y)
y = Dense(300)(y)
y = Activation('relu')(y)
#y = BatchNormalization()(y)
y = Dense(nb_actions, kernel_initializer=initializers.RandomUniform(minval=-0.003,  maxval=0.003))(y)
y = Activation('tanh')(y)
y = Lambda(lambda x: x * 20)(y)
#y = Activation('linear')(y)
actor = Model(inputs=[state_input], outputs=y)
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
#x = BatchNormalization()(x)
x = Dense(400)(x)
x = Activation('relu')(x)
#x = BatchNormalization()(x)
x = Dense(300)(x)
x = Activation('relu')(x)
#x = BatchNormalization()(x)
x = Dense(1, kernel_initializer=initializers.RandomUniform(minval=0.0003,   maxval=0.0003))(x)
x = Activation('linear')(x)

critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


def build_callbacks(env_name):
    checkpoint_weights_filename = 'results/Swimmer/ddpg_' + env_name + '_weights_{step}.h5f'
    log_filename = 'results/Swimmer/exp_9/ddpg_{}_log.json'.format(env_name)
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
GAMMA = 0.99

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1600000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, dt=0.005, mu=0., sigma=.25)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=GAMMA, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mse'])

callbacks = build_callbacks(ENV_NAME)
test_callbacks = build_test_callbacks(ENV_NAME)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
#agent.fit(env, nb_steps=1600000, visualize=False, callbacks=callbacks, verbose=1, gamma=GAMMA, nb_max_episode_steps=1600)

# After training is done, we save the final weights.
#agent.save_weights('results/Swimmer/exp_9/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
agent.load_weights('results/Swimmer/exp_9/ddpg_{}_weights.h5f'.format(ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.
history, state_history_nominal, episode_reward_nominal = agent.test(env, nb_episodes=1, visualize=True, action_repetition=1, callbacks=test_callbacks, nb_max_episode_steps=1600, \
                                                         initial_state=[0, 0, 0, 0,0,0,0,0,0,0], std_dev_noise=20, gamma=GAMMA)
u_max = 20
print(episode_reward_nominal, state_history_nominal)

'''
f = open("results/Swimmer/exp_4/data.txt", "a")



for i in frange(0.0, .12, 0.02):
    episode_reward_n = 0
    Var_n = 0
    terminal_mse = 0
    Var_terminal_mse = 0
    for j in range(n_samples):

        history, state_history, episode_reward = agent.test(env, nb_episodes=1, visualize=False, action_repetition=1, nb_max_episode_steps=800, initial_state=[0, 0, 0, 0,0,0,0,0,0,0], \
                                                    std_dev_noise=i*u_max, gamma=GAMMA)
        episode_reward_n += episode_reward
        Var_n += (episode_reward)**2
        terminal_mse += np.linalg.norm(state_history[800], axis=0)
        Var_terminal_mse += (np.linalg.norm(state_history[800], axis=0))**2

        #print(terminal_mse, state_history[30])
        #print(state_history, state_history[30]-[0, np.pi, 0, 0], np.linalg.norm(state_history[30]-[0, np.pi, 0, 0], axis=0))

    terminal_mse_avg = terminal_mse/n_samples
    episode_reward_n_avg = episode_reward_n/n_samples
    var_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_n - episode_reward_n**2)
    Var_terminal_mse_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_terminal_mse - terminal_mse**2)

    if var_avg > 0 :
        std_dev_avg = np.sqrt(var_avg)

    else:
        print("hii")
        std_dev_avg = 0
    std_dev_mse = np.sqrt(Var_terminal_mse_avg)

    f.write(str(i)+",\t"+str(terminal_mse_avg)+",\t"+str(std_dev_mse)+",\t"+str(episode_reward_n_avg)+",\t"+str(std_dev_avg)+"\n")
    print(terminal_mse, std_dev_mse, episode_reward_n_avg, std_dev_avg)
f.close()




i, x, y, z, a = np.loadtxt('results/Swimmer/exp_4/data.txt', dtype=np.float64, delimiter=',\t',  unpack=True, usecols=(0,1,2,3,4))
i = 100*i
plt.figure(1)
plt.plot(i,x,'orange', markersize=50)
plt.plot(i, (x+y), alpha=0.1, color='orange')
plt.plot(i, (x-y), alpha=0.1, color='orange')
plt.fill_between(i, (x+y), (x-y), alpha=0.3, color='orange')

plt.xlabel("% Of max. signal (Standard deviation of perturbed noise)")
plt.ylabel("Terminal mean squared error (Avergaed over {} samples)".format(n_samples))
plt.grid(linestyle='-')
plt.tight_layout()

plt.figure(2)
plt.xlabel("% Of max. signal (Standard deviation of perturbed noise)", fontsize=16)
plt.ylabel("Episodic reward fraction (Avg. over {} samples)".format(n_samples), fontsize=16)
plt.plot(i, z/episode_reward_nominal,'orange', markersize=50)
plt.plot(i, (z+a)/episode_reward_nominal, alpha=0.1, color='orange')
plt.plot(i, (z-a)/episode_reward_nominal, alpha=0.1, color='orange')
plt.fill_between(i, (z+a)/episode_reward_nominal, (z-a)/episode_reward_nominal, alpha=0.3, color='orange')
plt.grid(linestyle='-')
plt.tight_layout()

plt.show()
'''
