import numpy as np
from cartpole import agent 
from cartpole import env, GAMMA, STEPS_PER_EPISODE

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


u_max = 12
n_samples = 1000 # number of monte carlo samples

path_to_save = "results/Cartpole/exp_6/data.txt"
f = open(path_to_save, "a")


for i in frange(0.0, 1.02, 0.02):

    episode_reward_n = 0
    Var_n = 0
    terminal_mse = 0
    Var_terminal_mse = 0

    for j in range(n_samples):

        history, state_history, episode_reward = agent.test(env, nb_episodes=1, visualize=False, action_repetition=1, \
            nb_max_episode_steps=STEPS_PER_EPISODE, initial_state=[0, np.pi, 0, 0], std_dev_noise=i*u_max, gamma=GAMMA)
        episode_reward_n += episode_reward
        Var_n += (episode_reward)**2
        terminal_mse += np.linalg.norm(state_history[STEPS_PER_EPISODE], axis=0)
        Var_terminal_mse += (np.linalg.norm(state_history[STEPS_PER_EPISODE], axis=0))**2

    terminal_mse_avg = terminal_mse/n_samples
    episode_reward_n_avg = episode_reward_n/n_samples
    var_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_n - episode_reward_n**2)
    Var_terminal_mse_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_terminal_mse - terminal_mse**2)

    if var_avg > 0 :
        std_dev_avg = np.sqrt(var_avg)

    else:

        std_dev_avg = 0
    std_dev_mse = np.sqrt(Var_terminal_mse_avg)

    f.write(str(i)+",\t"+str(terminal_mse_avg)+",\t"+str(std_dev_mse)+",\t"+str(episode_reward_n_avg)+",\t"+str(std_dev_avg)+"\n")
    print(terminal_mse_avg, std_dev_mse, episode_reward_n_avg, std_dev_avg)
f.close()

