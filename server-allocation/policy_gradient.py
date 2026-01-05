import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from CloudComputing import ServerAllocationEnv

def moving_avg(scores, win_size):
    if len(scores) < win_size:
        return np.cumsum(scores) / np.arange(1, len(scores) + 1)
    return np.array([np.mean(scores[max(0, i - win_size):i+1]) for i in range(len(scores))])

def process_state(data):
    data = np.array(data, dtype=object)  
    type_map = {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}  
    one_hot = np.array([type_map[j] for j in data[:, 1]])  
    numeric_data = data[:, [0, 2, 3]].astype(float)  
    transformed_data = np.column_stack((numeric_data, one_hot, np.ones(len(data))))  
    return np.mean(transformed_data, axis=0).flatten()


class DataScaler:
    def __init__(self, dim, eps=1e-8):
        self.count = 0
        self.mu = np.zeros(dim)
        self.var = np.zeros(dim)
        self.eps = eps

    def update(self, x):
        self.count += 1
        delta = x - self.mu
        self.mu += delta / self.count
        delta2 = x - self.mu
        self.var += delta * delta2

    def normalize(self, x):
        if self.count < 2:
            std_dev = np.sqrt(self.eps)
        else:
            std_dev = np.sqrt(self.var / (self.count - 1)) + self.eps
        return (x - self.mu) / std_dev

class Policy(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(in_dim, hid_dim)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(hid_dim, out_dim)
        self.probs = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return self.probs(x)

num_iters = 1000
lr = 2e-3
h_size = 16
rolling_win = 500
b_alpha = 0.01

env = ServerAllocationEnv()
feat_size = 7
actions = 8

agent = Policy(feat_size, h_size, actions)
optimizer = optim.Adam(agent.parameters(), lr=lr)

scaler = DataScaler(dim=feat_size)
b = 0.0
reward_log = []
#epsilon = 0.1

for it in range(num_iters):
    obs, _ = env.reset()
    end_flag = False
    
    while not end_flag:
        epsilon =  max(0.01, 0.9 * (0.99 ** it))  # Faster decay
        z = process_state(obs)
        scaler.update(z)
        norm_z = scaler.normalize(z)
        state = torch.FloatTensor(norm_z).unsqueeze(0)
        
        act_probs = agent(state)
        dist = torch.distributions.Categorical(act_probs)
        act_id = dist.sample()
        log_p = dist.log_prob(act_id)
        
        #action = int(act_id.item()) + 1
          # Add exploration parameter

        if np.random.uniform() < epsilon:
            action = np.random.randint(1, actions + 1)  # Exploration
        else:
            action = int(act_id.item()) + 1  # Exploitation

        new_obs, rwd, _, end_flag, _ = env.step(action)
        
        reward_log.append(rwd)
        
        b = b + b_alpha * (rwd - b)
        
        loss = -log_p * (rwd - b)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        obs = new_obs
        
    if len(reward_log) < rolling_win:
        avg_rwd = np.mean(reward_log)
    else:
        avg_rwd = np.mean(reward_log[-rolling_win:])

    print(f"Step {len(reward_log)}: Avg Reward = {avg_rwd:.2f}")

avg_rewards = moving_avg(reward_log, rolling_win)
plt.figure(figsize=(8, 5))
plt.plot(avg_rewards, label=f"Window Avg (win_size={rolling_win})", color='r')
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Window Avg Reward (Policy Gradient)")
plt.legend()
plt.show()

print("Final avg reward (:", np.average(reward_log[-100:]))