import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import pandas as pd

# Initialize environment
env = gym.make("MountainCar-v0")
epsilon = 0.2  # Start with high exploration
gamma = 0.99  # Discount factor

# Tile coding parameters
tile_number = 8
tile_size = 10

alpha = 0.1 / tile_number # Learning rate

Model = 'Expected_Sarsa'

class TileCoder:
    def __init__(self, tile_number, tile_size):
        self.tile_size = tile_size
        self.tile_number = tile_number

    def GetFeatures(self, state, variables):
        tiles = []
        for i in range(0, self.tile_number):
            offset_coefficient = i/(self.tile_number * 4)
            first_range = (variables.high[0] - variables.low[0])
            second_range = (variables.high[1] - variables.low[1])
            first_offset_value = first_range * offset_coefficient
            second_offset_value = second_range * offset_coefficient
            scale_first = (state[0] - (variables.low[0] + i * first_offset_value)) / first_range
            scale_second = (state[1] - (variables.low[1] + i * second_offset_value)) / second_range

            first_index = int(scale_first * self.tile_size)
            second_index = int(scale_second * self.tile_size)

            mid_index = first_index * self.tile_size + second_index

            final_index = mid_index + i * self.tile_size * self.tile_size

            tiles.append(final_index)

        return tiles


tile_coder = TileCoder(tile_number, tile_size)

# Initialize weights

# Training episodes
n_episodes = 200
replications = 10
steps_per_episode = pd.DataFrame(np.zeros((n_episodes, replications)))

for j in range(0, replications):
    w = np.zeros((tile_size * tile_size * 3 * tile_number,))  # 3 actions, tiles per feature

    for i in range(0, n_episodes):
        state, _ = env.reset()
        done = False
        total_step = 0

        while (not done):
            tiles = tile_coder.GetFeatures(state, env.observation_space)

            rand = random.random()
            if rand <= epsilon:
                action = random.choice([0, 1, 2])
            else:
                q_values = []
                zero_count = 0
                for a in range(3):
                    q_value = w[np.array(tiles) + a * tile_size * tile_size * tile_number].sum()
                    q_values.append(q_value)
                    if q_value == 0:
                        zero_count += 1

                if zero_count <= 1:
                    action = np.argmax(q_values)
                else:
                    max_indices = np.argwhere(q_values == np.max(q_values)).flatten()

                    # Select one index randomly
                    action = np.random.choice(max_indices)

            # Take action
            new_state, reward, done, _, _ = env.step(action)
            if done:
                q_value = w[np.array(tiles) + action * tile_size * tile_size * tile_number].sum()
                w[np.array(tiles) + action * tile_size * tile_size * tile_number] += alpha * (reward - q_value)
                steps_per_episode[j].iloc[i] = total_step
            else:
                q_values_new = []
                new_tiles = tile_coder.GetFeatures(new_state, env.observation_space)
                for a in range(3):
                    q_value = w[np.array(new_tiles) + a * tile_size * tile_size * tile_number].sum()
                    q_values_new.append(q_value)
                if Model == 'Q_Learning':
                    q_hat_value = np.max(q_values_new)
                elif Model == 'Expected_Sarsa':
                    def softmax(x):
                        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
                        return exp_x / exp_x.sum()

                    probabilities = softmax(np.array(q_values_new))
                    q_hat_value = (probabilities * np.array(q_values_new)).sum()

                q_value = w[np.array(tiles) + action * tile_size * tile_size * tile_number].sum()
                w[np.array(tiles) + action * tile_size * tile_size * tile_number] += alpha * (
                            reward + gamma * q_hat_value - q_value)

            state = new_state
            total_step += 1

            if (total_step + 1) % 100 == 0:
                print(f"Episode {total_step + 1}: Steps = {total_step}")


# plt.rcParams['font.family'] = 'Times New Roman'
#
# steps_per_episode_mean = steps_per_episode.mean(axis=1)
# # steps_per_episode_mean_Qlearning = steps_per_episode_mean.copy()
# plt.plot((steps_per_episode_mean), label = 'Expected_Sarsa')
# plt.plot((steps_per_episode_mean_Qlearning), label ='Q-Learning')
#
# plt.title("Steps per Episode")
# plt.xlabel("Episode")
# plt.ylabel("Steps (Averaged on Ten Runs)")
# plt.legend()
# plt.savefig('Mountain Car.png', dpi= 1200)
#
# plt.show()
