# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:15:04 2025

@author: hbiru
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GymTrafficEnv(gym.Env):
    def __init__(self):
        self.alpha       = [0.28, 0.4]    # Arrival probabilities
        self.max_queue   = 1810          # 1800 arrivals + 10 initial
        self.max_red     = 10            # Minimum-red hold (seconds)
        self.green_depart = 0.9          # Departure probability when green
        self.max_time    = 1800          # Episode length in time slots

        self.action_space = spaces.Discrete(2)   # 0=keep, 1=switch
        self.observation_space = spaces.MultiDiscrete([
            self.max_queue + 1,  # r1: 0…1810
            self.max_queue + 1,  # r2: 0…1810
            2,                   # g: which road is green (0 or 1)
            self.max_red + 1     # t_since_red: 0…10
        ])

    def reset(self):
        # Initial queues ∼ Uniform{0,…,10}, starting with road 0 green
        r1 = np.random.randint(0, 11)
        r2 = np.random.randint(0, 11)
        g = 0
        t_since_red = 0
        self.state = np.array([r1, r2, g, t_since_red], dtype=int)
        self.t = 0
        return self.state, {}

    def step(self, action):
        r1, r2, g, t_since_red = self.state

        # Enforce 10s hold: override any switch if red<10
        if action == 1 and t_since_red < self.max_red:
            action = 0

        # Phase update
        if action == 1:
            g_new = 1 - g
            t_since_red_new = 0
        else:
            g_new = g
            t_since_red_new = min(t_since_red + 1, self.max_red)

        # Departure probabilities for green vs red
        if g_new == 0:
            d1_prob = self.green_depart
            d2_prob = self.green_depart * (1 - t_since_red_new**2 / 100)
        else:
            d2_prob = self.green_depart
            d1_prob = self.green_depart * (1 - t_since_red_new**2 / 100)

        # Sample arrivals and departures
        a1 = np.random.rand() < self.alpha[0]
        a2 = np.random.rand() < self.alpha[1]
        d1 = np.random.rand() < d1_prob
        d2 = np.random.rand() < d2_prob

        # Update queues (clip to [0,1810])
        r1_new = int(np.clip(r1 - d1 + a1, 0, self.max_queue))
        r2_new = int(np.clip(r2 - d2 + a2, 0, self.max_queue))
        self.state = np.array([r1_new, r2_new, g_new, t_since_red_new], dtype=int)

        # Reward is negative total queue length
        reward = - (r1_new + r2_new)

        # Time bookkeeping
        self.t += 1
        terminated = False
        truncated  = (self.t >= self.max_time)

        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass