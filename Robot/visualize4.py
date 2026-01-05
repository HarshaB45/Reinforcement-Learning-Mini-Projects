# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 02:11:46 2025

@author: hbiru
"""

import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3 import SAC   # If you trained a PPO model you have to import PPO here. I hope you get the gist.

class ModifiedRobotNavigationEnv(gym.Wrapper):
    def __init__(self, env, H):
        super().__init__(env)
        self.H = H   # Number of time slots in a mini episode.
        
        # The action space of the modified robot navigation environment.
        Delta = H*self.env.delta
        self.action_space = gym.spaces.Box(-Delta*np.ones(2), Delta*np.ones(2), dtype=np.float32)
        
    
    def conventional_policy(self, robot_position, goal_intermediate):
        
        tol = 1e-5
        x_current, y_current = robot_position
        x_goal, y_goal = goal_intermediate
        
        if x_goal - x_current > tol:
            return 3  # go right
        elif x_goal - x_current < -tol:
            return 2  # go left
        elif y_goal - y_current > tol:
            return 0  # go up
        elif y_goal - y_current < -tol:
            return 1  # go down
        else:
            return -1  # don't move




    def step(self, action):
        # Compute the intermediate goal
        goal_intermediate = self.env.robot_position + action        
        grid_x = int(goal_intermediate[0]/self.env.delta) + 1
        grid_y = int(goal_intermediate[1]/self.env.delta) + 1        
        grid_x = self.env.delta*(0.5 + (grid_x - 1))
        grid_y = self.env.delta*(0.5 + (grid_y - 1))
        goal_intermediate = np.array([grid_x, grid_y])
        
        # Simulate a mini-episode using the conventional policy.
        reward_miniepisode = 0
        for h in range(self.H):
            reward = -np.sqrt(np.sum((self.env.robot_position - self.env.goal)**2))
            
            a = self.conventional_policy(self.env.robot_position, goal_intermediate)  # Call conventional policy.
            
            if a!=-1: # If a==-1, then robot position does not change.
                self.env.robot_position = self.env.robot_position + self.env.action_dict[a+0]*self.env.delta
            
            self.env.trail.append(self.env.robot_position)
            
            # Check for collision
            terminated = self.env.check_collision()
            if terminated:
                reward = -10000
            
            # Check if goal is reached
            if not(terminated):
                if (self.env.goal[0] - self.env.robot_position[0])**2 + (self.env.goal[1] - self.env.robot_position[1])**2<=self.env.goal_radius**2:
                    terminated = True
                    reward = 1000
            
            reward_miniepisode+=reward
            
            self.env.t += 1
            truncated = False
            if self.env.t>self.env.Horizon:
                truncated = True
            
            if self.env.render_mode == "human":
                self.env.render()
            
            if terminated or truncated:
                break
        
        self.env.observation = np.concatenate((self.env.get_lidar_reading(),self.env.robot_position))
        
        return self.env.observation, reward_miniepisode, terminated, truncated, {}

# Load the trained model
model = SAC.load("MODEL4")   # If you used PPO the you have to use PPO.load("MODEL3"). I hope you get the gist.

# Initiate the robot navigation environment.
env = RobotNavigationEnv(render_mode='human')
# env = RobotNavigationEnv()
H = 20 # This MUST be same as the one used during training.
env = ModifiedRobotNavigationEnv(env, H)

# Reset environment
x, _ = env.reset()
terminated = False
truncated = False
total_reward = 0

while not (terminated or truncated):
    a, _states = model.predict(x, deterministic=True)  # a is an array of continuous actions
    x, r, terminated, truncated, _ = env.step(a)  # Pass full continuous action vector here
    total_reward += r

print('Sum of reward = {}'.format(total_reward))

env.render()