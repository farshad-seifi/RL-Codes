import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class cliff_walking:
    def __init__(self):
        self.columns_number = 12
        self.rows_number = 4
        self.start_point = [0, 3]
        self.final_point = [11, 3]
        self.cliff = [[1,3], [2,3], [3,3], [4,3], [5,3], [6,3], [7,3], [8,3], [9,3], [10,3]]
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 1
        self.current_state = self.start_point
        self.Q_values = np.zeros((4, self.columns_number, self.rows_number))
        self.actions_list = ['u', 'd', 'r', 'l']

    def select_action(self):
        rand = np.random.rand()
        if rand <= self.epsilon:
            action = np.random.choice(self.actions_list)
        else:
            x = self.current_state[0]
            y = self.current_state[1]
            l = np.array([self.Q_values[0][x][y], self.Q_values[1][x][y], self.Q_values[2][x][y], self.Q_values[3][x][y]])
            action = self.actions_list[l.argmax()]
        return action

    def select_action_max(self):
        x = self.current_state[0]
        y = self.current_state[1]
        l = np.array([self.Q_values[0][x][y], self.Q_values[1][x][y], self.Q_values[2][x][y], self.Q_values[3][x][y]])
        action = self.actions_list[l.argmax()]
        return action

    def next_state(self, action):
        x = self.current_state[0]
        y = self.current_state[1]
        if action == 'u':
            y -= 1
        elif action == 'd':
            y += 1
        elif action == 'r':
            x += 1
        elif action == 'l':
            x -= 1

        y = max(y, 0)
        y = min(y, self.rows_number - 1)
        x = max(0, x)
        x = min(x,self.columns_number - 1)

        return [x, y]

    def reward_calculator(self, next_state):
        x = next_state[0]
        y = next_state[1]
        if next_state == self.final_point:
            reward = 1
            is_terminate_state = True
        elif [x,y] in self.cliff:
            reward = -100
            is_terminate_state = True
        else:
            reward = -1
            is_terminate_state = False

        return reward, is_terminate_state

    def reset_game(self):
        self.current_state = self.start_point

    def Q_value_updator(self, current_state, action, reward, next_state, next_action):
        action_num = self.actions_list.index(action)
        next_action_num = self.actions_list.index(next_action)
        x = current_state[0]
        y = current_state[1]
        next_x = next_state[0]
        next_y = next_state[1]
        self.Q_values[action_num][x][y] = self.Q_values[action_num][x][y] +\
                                          self.alpha * (reward + self.gamma * self.Q_values[next_action_num][next_x][next_y] - self.Q_values[action_num][x][y])

simulations = 15000
episode_count_sara = pd.DataFrame(np.zeros((simulations,2)))
w = cliff_walking()
Total_Reward = 0
Reward_Sarsa = pd.DataFrame(np.zeros((500, 10)))

for j in range(0, 10):
    episode_count_sara = pd.DataFrame(np.zeros((simulations, 2)))
    w = cliff_walking()
    Total_Reward = 0
    for i in range(0, simulations):
        current_state = w.current_state
        action = w.select_action()
        next_state = w.next_state(action)
        reward, is_terminate_state = w.reward_calculator(next_state)
        Total_Reward += reward
        if is_terminate_state:
            w.reset_game()
            episode_count_sara[0].iloc[i] = episode_count_sara[0].iloc[i - 1] + 1
            episode_count_sara[1].iloc[i] = reward
            Total_Reward = 0
            w.Q_values[w.actions_list.index(action)][current_state[0]][current_state[1]] = reward * w.gamma
        else:
            w.current_state = next_state
            next_action = w.select_action()
            w.Q_value_updator(current_state, action, reward, next_state, next_action)
            if i >= 1:
                episode_count_sara[0].iloc[i] = episode_count_sara[0].iloc[i - 1]

    episode_count_sara = episode_count_sara[episode_count_sara[1] != 0]
    episode_count_sara.reset_index(inplace=True, drop=True)
    episode_count_sara[2] = 0
    for o in range(1, len(episode_count_sara) + 1):
        episode_count_sara[2].iloc[o - 1] = episode_count_sara[1].iloc[:o].mean()
    Reward_Sarsa[j] = episode_count_sara[2].iloc[:500]

simulations = 15000
w = cliff_walking()
Total_Reward = 0
Reward_Q_Learning = pd.DataFrame(np.zeros((500, 10)))

for j in range(0, 10):
    episode_count_QLearning = pd.DataFrame(np.zeros((simulations, 2)))
    w = cliff_walking()
    Total_Reward = 0
    for i in range(0, simulations):
        current_state = w.current_state
        action = w.select_action()
        next_state = w.next_state(action)
        reward, is_terminate_state = w.reward_calculator(next_state)
        Total_Reward += reward

        if is_terminate_state:
            w.reset_game()
            episode_count_QLearning[0].iloc[i] = episode_count_QLearning[0].iloc[i-1] + 1
            episode_count_QLearning[1].iloc[i] = reward
            Total_Reward = 0
            w.Q_values[w.actions_list.index(action)][current_state[0]][current_state[1]] = reward * w.gamma
        else:
            w.current_state = next_state
            next_action = w.select_action_max()
            w.Q_value_updator(current_state, action, reward, next_state, next_action)
            if i >= 1:
                episode_count_QLearning[0].iloc[i] = episode_count_QLearning[0].iloc[i-1]

    episode_count_QLearning = episode_count_QLearning[episode_count_QLearning[1] != 0]
    episode_count_QLearning.reset_index(inplace=True, drop=True)
    episode_count_QLearning[2] = 0
    for o in range(1, len(episode_count_QLearning) + 1):
        episode_count_QLearning[2].iloc[o - 1] = episode_count_QLearning[1].iloc[:o].mean()
    Reward_Q_Learning[j] = episode_count_QLearning[2].iloc[:500]


R_Q = Reward_Q_Learning.mean(axis= 1)
R_S = Reward_Sarsa.mean(axis= 1)


plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(10, 6))
plt.plot(R_S.index, R_S, color='blue', label="Sarsa")
plt.plot(R_Q.index, R_Q, color='red', label="Q-Learning")


# Adding labels and title
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Results of Sarsa and Q-Learning applied to the cliff-walking task (Ten simulations)')

# Adding a legend to differentiate the lines
plt.legend()

# plt.savefig('Cliff.png', dpi= 1200)
# Displaying the plot
plt.show()
