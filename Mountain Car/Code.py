import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import pandas as pd

# Initialize environment
env = gym.make("MountainCar-v0")
epsilon = 1.0  # Start with high exploration
epsilon_decay = 0.99  # Decay rate for epsilon
min_epsilon = 0.1  # Minimum exploration rate
gamma = 0.99  # Discount factor
alpha = 0.01  # Learning rate

# Tile coding parameters
tile_number = 9
tile_size = 8

class TileCoder:
    def __init__(self, tile_number, tile_size):
        self.tile_number = tile_number
        self.tile_size = tile_size

    def get_features(self, state, variables):
        # Normalize state
        state_scaled = (state - variables.low) / (variables.high - variables.low)
        tiles = []
        for i in range(self.tile_number):
            offset = i / self.tile_number
            state_with_offset = (state_scaled + offset) % 1
            tile_index = (state_with_offset * self.tile_size).astype(int)
            tiles.append(tile_index[0] * self.tile_size + tile_index[1] + i * self.tile_size * self.tile_size)
        return tiles


tile_coder = TileCoder(tile_number, tile_size)

# Initialize weights
weights = np.zeros((tile_size * tile_size * 3 * tile_number,))  # 3 actions, tiles per feature

# Training episodes
n_episodes = 200
steps_per_episode = []

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_steps = 0

    while not done:
        # Get tile features
        tiles = tile_coder.get_features(state, env.observation_space)

        # Choose action (epsilon-greedy)
        if random.random() < epsilon:
            action = env.action_space.sample()  # Random action
        else:
            q_values = []
            for a in range(3):
                q_value = sum(
                    weights[int(tile) + a * tile_size * tile_size * tile_number] for tile in tiles
                )
                q_values.append(q_value)
            action = np.argmax(q_values)

        # Take action
        new_state, reward, done, _, _ = env.step(action)

        # Get next tile features
        new_tiles = tile_coder.get_features(new_state, env.observation_space)

        # Compute target and update weights
        if done:
            target = reward  # No future reward if episode ends
        else:
            next_q_values = []
            for a in range(3):
                q_value = sum(weights[int(tile) + a * tile_size * tile_size * tile_number] for tile in new_tiles)
                next_q_values.append(q_value)
            target = reward + gamma * max(next_q_values)

        current_q = sum(weights[int(tile) + action * tile_size * tile_size * tile_number] for tile in tiles)
        td_error = target - current_q

        for tile in tiles:
            weights[int(tile) + action * tile_size * tile_size * tile_number] += alpha * td_error

        # Update state
        state = new_state
        total_steps += 1

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Track steps for analysis
    steps_per_episode.append(total_steps)

    # Logging
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}: Steps = {total_steps}")

# Plot results
plt.rcParams['font.family'] = 'Times New Roman'

plt.plot(np.log(steps_per_episode))
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps (Log Scale)")
plt.savefig('Mountain Car.png', dpi= 1200)

plt.show()
