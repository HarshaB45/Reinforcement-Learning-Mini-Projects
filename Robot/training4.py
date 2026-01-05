import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC

class ModifiedRobotNavigationEnv(gym.Wrapper):
    def __init__(self, env, H):
        super().__init__(env)
        self.H = H # Number of time slots in a mini episode.
        
        # The action space of the modified robot navigation environment.
        Delta = H * self.env.delta
        self.action_space = gym.spaces.Box(-Delta * np.ones(2), Delta * np.ones(2), dtype=np.float32)

    def conventional_policy(self, robot_position, goal_intermediate):
        tol = 1e-5
        x_current, y_current = robot_position
        x_goal, y_goal = goal_intermediate
        if x_goal - x_current > tol:
            return 3
        elif x_goal - x_current < -tol:
            return 2
        elif y_goal - y_current > tol:
            return 0
        elif y_goal - y_current < -tol:
            return 1
        else:
            return -1

    def step(self, action):
        # Compute the intermediate goal
        goal_intermediate = self.env.robot_position + action
        grid_x = int(goal_intermediate[0] / self.env.delta) + 1
        grid_y = int(goal_intermediate[1] / self.env.delta) + 1
        grid_x = self.env.delta * (0.5 + (grid_x - 1))
        grid_y = self.env.delta * (0.5 + (grid_y - 1))
        goal_intermediate = np.array([grid_x, grid_y])
        
        # Simulate a mini-episode using the conventional policy.
        reward_miniepisode = 0
        truncated = False

        for _ in range(self.H):
            reward = -np.linalg.norm(self.env.robot_position - self.env.goal) # Call conventional policy.
            a = self.conventional_policy(self.env.robot_position, goal_intermediate)
            if a != -1: # If a==-1, then robot position does not change.
                self.env.robot_position += self.env.action_dict[a] * self.env.delta

            self.env.trail.append(self.env.robot_position)
            
            # Check for collision
            terminated = self.env.check_collision()
            if terminated:
                reward = -10000
            
            # Check if goal is reached
            elif np.linalg.norm(self.env.robot_position - self.env.goal) <= self.env.goal_radius:
                reward = 1000
                terminated = True

            reward_miniepisode += reward
            self.env.t += 1
            truncated = self.env.t > self.env.Horizon

            if self.env.render_mode == "human":
                self.env.render()

            if terminated or truncated:
                break

        self.env.observation = np.concatenate((self.env.get_lidar_reading(), self.env.robot_position))
        return self.env.observation, reward_miniepisode, terminated, truncated, {}



# You can copy-paste the code for the custom callback LoggingAndSavingCallback
# that you wrote for training3.py. All you need to change is the code for
# initiating the environment during testing.
class LoggingAndSavingCallback(BaseCallback):
    def __init__(self, test_period, test_count, verbose=0):
        super().__init__(verbose)
        self.test_period = test_period
        self.test_count = test_count
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.training_log = []
        self.testing_log = []
        self.best_avg_reward = -np.inf

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]

        if self.locals['dones'][0]:
            self.training_log.append(self.current_episode_reward)
            np.save('training_log.npy', np.array(self.training_log))
            self.current_episode_reward = 0

        if self.num_timesteps > 0 and self.num_timesteps % self.test_period == 0:
            self.model.save("LATEST_MODEL")

            test_env = ModifiedRobotNavigationEnv(RobotNavigationEnv(), H)
            test_rewards = []

            for _ in range(self.test_count):
                obs, _ = test_env.reset()
                done = False
                episode_reward = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = test_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated

                test_rewards.append(episode_reward)

            test_env.close()
            avg_reward = np.mean(test_rewards)
            self.testing_log.append(avg_reward)
            np.save('testing_log.npy', np.array(self.testing_log))

            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.model.save("BEST_MODEL")

        return True

            
# Initiate the robot navigation environment.
env = RobotNavigationEnv()
H = 20 # H is the duration of a mini-episode. My advide is to change it between 10 to 40. But you can go crazy with it!
env = ModifiedRobotNavigationEnv(env, H)

# Initiate an instance of the LoggingAndSavingCallback. Desription of test_period
# and test_count are there in _init__ function of LoggingAndSavingCallback.
test_period = 5000 # Default value. You can change it.
test_count = 10 # Default value. You can change it.
callback = LoggingAndSavingCallback(test_period, test_count)

# The code that you use to train the RL agent for the robot navigation environment
# goes below this line. The total number of lines is unlikely to be more than 10.
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=4e-4,           # Lower LR for smoother training
    batch_size=256,               # Increase batch size from default (usually 64 or 128)
    buffer_size=100_000,          # Large replay buffer (default is usually 1e6 but can start smaller)
    train_freq=1,                 # Train every step (default)
    gradient_steps=1,             # Update model after each rollout step
    ent_coef="auto_0.1",          # Fix entropy coefficient to a small value to stabilize exploration
    target_entropy='auto',        # You can keep auto, but sometimes fixed ent_coef helps stability
    gamma=0.99,                   # Default discount factor
    tau=0.005,                    # Target smoothing coefficient, default
    policy_kwargs=dict(net_arch=[128, 128]),  # Your 2-layer MLP with 256 units each
)

model.learn(total_timesteps=500_000, callback=callback)

# Close the robot navigation environment.
env.close()


# Write just ONE line of code below to save the model that you have trained.
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL4.
model.save("MODEL4")
