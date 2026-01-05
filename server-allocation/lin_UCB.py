import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym 
from CloudComputing import ServerAllocationEnv

def receding_window_avg(reward_arr, window_size):
    if len(reward_arr) < window_size:
        return np.cumsum(reward_arr) / np.arange(1, len(reward_arr) + 1)
    return np.array([np.mean(reward_arr[max(0, i - window_size):i+1]) for i in range(len(reward_arr))])

def aggregateContext(obsv):
    obsv = np.array(obsv)
    job_type_map = {'A': 0, 'B': 1, 'C': 2}
    obsv[:, 1] = np.vectorize(job_type_map.get)(obsv[:, 1])
    obsv = obsv.astype(float)
    obsv_with_bias = np.concatenate((obsv, np.ones((obsv.shape[0], 1))), axis=1)
    z = np.mean(obsv_with_bias, axis=0).flatten()  
    return z

def LinUCBPolicy(env):
    Nactions = 8
    Nfeatures = 4
    theta = np.zeros((Nactions, Nfeatures + 1))
    A = np.zeros((Nactions, Nfeatures + 1, Nfeatures + 1))
    b = np.zeros((Nactions, Nfeatures + 1))
    alpha = 2.0
    Nepisodes = 100

    action_list = []
    reward_arr = []

    for episode in range(Nepisodes):
        print(f"Running episode {episode + 1} of {Nepisodes}")
        obsv, _ = env.reset()
        truncated = False
        while not truncated:
            z = aggregateContext(obsv)
            scores = []
            for a in range(Nactions):
                A_reg = A[a] + 0.01 * np.eye(Nfeatures + 1)
                est_reward = np.dot(theta[a], z)
                uncertainty = alpha * np.sqrt(np.dot(z, np.linalg.solve(A_reg, z)))
                scores.append(est_reward + uncertainty)
            scores = np.array(scores)
            action = np.argmax(scores) + 1
            action_list.append(action)
            obsv_next, reward, _, truncated, _ = env.step(action)
            reward_arr.append(reward)
            if not truncated:
                z_temp = z.reshape(-1, 1)
                A[action - 1] += np.dot(z_temp, z_temp.T)
                b[action - 1] += reward * z
                A_reg = A[action - 1] + 0.01 * np.eye(Nfeatures + 1)
                theta[action - 1] = np.linalg.solve(A_reg, b[action - 1])
                obsv = obsv_next
    return np.array(reward_arr), theta

if _name_ == "_main_":
    env = ServerAllocationEnv()
    reward_arr, theta = LinUCBPolicy(env)
    window_size = 500
    time_avg_rewards = receding_window_avg(reward_arr, window_size)
    plt.figure(figsize=(8, 5))
    plt.plot(time_avg_rewards, label=f"Receding Window Avg (window={window_size})", color='r')
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward")
    plt.title("Receding Window Time Average of Rewards - LinUCB")
    plt.legend()
    plt.ylim([-50, -20])
    plt.show()
    print("Reward array:", reward_arr)
    print("Average reward over last 100 time steps:", np.average(reward_arr[-100:]))