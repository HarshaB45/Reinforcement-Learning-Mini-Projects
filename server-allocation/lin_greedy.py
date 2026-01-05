import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from trial import ServerAllocationEnv

def receding_window_avg(reward_arr, window_size):
    if len(reward_arr) < window_size:
        return np.cumsum(reward_arr) / np.arange(1, len(reward_arr) + 1)
    
    smoothed_rewards = []
    for i in range(len(reward_arr)):
        start_index = max(0, i - window_size)
        window = reward_arr[start_index : i + 1]
        smoothed_rewards.append(np.mean(window))
    
    return np.array(smoothed_rewards)

def actionDistribution(action_list):
    
    plt.figure(figsize=(8, 5))
    action_labels = np.arange(1, 9)
    action_counts = [action_list.count(a) for a in action_labels]
    
    plt.bar(action_labels, action_counts, color='b', alpha=0.7)
    plt.xlabel("Action Values")
    plt.ylabel("Frequency")
    plt.title("Action Distribution")
    plt.xticks(action_labels)
    plt.show()
    
def aggregateContext(obsv):
    obsv = np.array(obsv)
    categories = {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}
    encoded_list = []

    for row in obsv:
        new_tuple = (row[0], *categories[row[1]], row[2], row[3], 1)  # Encoding categorical features
        encoded_list.append(new_tuple)

    encoded_array = np.array(encoded_list, dtype=float)
    num_indices = [0, 4, 5]
    numerical_means = np.mean(encoded_array[:, num_indices], axis=0)
    one_hot_values = encoded_array[0, 1:4]  
    bias = encoded_array[0, -1]  
    
    return np.concatenate([numerical_means, one_hot_values, [bias]])

def LinGreedyPolicy(env):   
    Nactions = 8  
    Nfeatures = 4  
    
    A = np.random.normal(0, 0.01, (Nactions, Nfeatures+3, Nfeatures+3))
    b = np.random.normal(0, 0.01, (Nactions, Nfeatures+3))
    theta = np.random.normal(0, 0.01, (Nactions, Nfeatures+3))
    
    epsilon = 0.9  
    Nepisodes = 1000  

    reward_arr = []
    priority_list = []
    processing_time_list = []
    num_jobs_list = []
    allocated_servers_list = []
    action_list = []
    for n in range(Nepisodes):
        epsilon = max(0.01, 0.9 * (0.995 ** n))  
        obsv, _ = env.reset()
        truncated = False
        while not truncated:
            z = aggregateContext(obsv)  
            v = np.random.uniform()
            action = np.random.randint(1, Nactions+1) if v <= epsilon else np.argmax(theta @ z) + 1
            
            action_list.append(action)
            actionDistribution(action)
            obsv_next, reward, _, truncated, _ = env.step(action)  
            reward_arr.append(reward)
            
            num_jobs = len(obsv)
            priorities = [job[0] for job in obsv]
            processing_times = [job[3] for job in obsv]
            
            num_jobs_list.append(num_jobs)
            priority_list.append(np.mean(priorities))  
            processing_time_list.append(np.mean(processing_times))  
            allocated_servers_list.append(action)
            
            if not truncated:
                z_temp = z.reshape((-1, 1))
                A[action - 1] += z_temp @ z_temp.T  
                b[action - 1] += reward * z_temp.reshape(-1)  
                theta[action - 1] = np.matmul(np.linalg.inv(A[action - 1] + 0.01*np.eye(Nfeatures+3)), b[action - 1])
                obsv = obsv_next  
                z = aggregateContext(obsv)  
    
    return np.array(reward_arr), priority_list, processing_time_list, num_jobs_list, allocated_servers_list

def plot_results(priority_list, processing_time_list, num_jobs_list, allocated_servers_list):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=priority_list, y=allocated_servers_list)
    plt.xlabel("Average Priority (Lower is Higher Priority)")
    plt.ylabel("Allocated Servers")
    plt.title("Priority vs Allocated Servers")
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=processing_time_list, y=allocated_servers_list)
    plt.xlabel("Average Estimated Processing Time")
    plt.ylabel("Allocated Servers")
    plt.title("Processing Time vs Allocated Servers")
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=num_jobs_list, y=allocated_servers_list)
    plt.xlabel("Number of Jobs in Time Slot")
    plt.ylabel("Allocated Servers")
    plt.title("Number of Jobs vs Allocated Servers")
    
    plt.tight_layout()
    plt.show()

# Initialize environment and execute policy
env = ServerAllocationEnv()
reward_arr, priority_list, processing_time_list, num_jobs_list, allocated_servers_list = LinGreedyPolicy(env)

# Compute and plot receding window average
window_size = 500
time_avg_rewards = receding_window_avg(reward_arr, window_size)

plt.figure(figsize=(8, 5))
plt.plot(time_avg_rewards, label=f"Receding Window Avg (window={window_size})", color='r')
plt.xlabel("Time Steps")
plt.ylabel("Average Reward")
plt.title("Receding Window Time Average of Rewards")
plt.legend()
plt.show()

# Generate required graphs
plot_results(priority_list, processing_time_list, num_jobs_list, allocated_servers_list)