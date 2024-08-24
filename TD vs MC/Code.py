import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

class environment:
    def __init__(self):
        self.actions = ['R', 'L']
        self.states = [-3, -2, -1 ,0 ,1 , 2, 3]
        self.current_state = 0

    def next_state(self, action):
        if action == 'R':
            self.current_state += 1
        elif action == 'L':
            self.current_state -= 1

    def check_terminate(self):
        if self.current_state == 3:
            reward = 1
            is_terminate= True
        elif self.current_state == -3:
            reward = -1
            is_terminate = True
        else:
            reward = 0
            is_terminate = False

        return reward, is_terminate

    def reset_game(self, is_terminate):
        if is_terminate:
            self.current_state = 0

    def select_action(self):
        if np.random.rand() > 0.5:
            action = 'R'
        else:
            action = 'L'

        return action

actual_state_values = [-2/3, -1/3, 0, 1/3, 2/3]
actual_state_values = np.array(actual_state_values)
s = environment()
states_list = [0]
state_value_MC = np.zeros((5,4))
state_value_MC = pd.DataFrame(state_value_MC)
state_value_MC[0] = s.states[1:-1]
gamma = 1
N_walks = 1000
n_simulation = 100
MSE_MC = np.zeros((N_walks, n_simulation))
MSE_MC = pd.DataFrame(MSE_MC)
MSE = np.zeros((N_walks, 2))
MSE = pd.DataFrame(MSE)

for q in range(0, n_simulation):
    s = environment()
    states_list = [0]
    state_value_MC = np.zeros((5, 4))
    state_value_MC = pd.DataFrame(state_value_MC)
    state_value_MC[0] = s.states[1:-1]
    for i in range(0, N_walks):
        action = s.select_action()
        s.next_state(action)
        reward, is_terminate = s.check_terminate()
        states_list.append(s.current_state)

        if is_terminate:
            states_list.pop()
            l = len(states_list)
            for j in range(0, l):
                updated_value = (gamma ** (j+1)) * reward
                state_value_MC[1].iloc[states_list[-(j+1)] + 2] += updated_value
                state_value_MC[2].iloc[states_list[-(j+1)] + 2] += 1
            s.reset_game(is_terminate)
            states_list = [0]

        state_value_MC[3] = state_value_MC[1] / (state_value_MC[2] + 0.00000001)
        MSE_MC[q].iloc[i] = ((state_value_MC[3] - actual_state_values) ** 2).sum()/5

MSE[0] = MSE_MC.mean(axis= 1)

s = environment()
state_value = np.zeros((7,4))
state_value = pd.DataFrame(state_value)
state_value[0] = s.states
gamma = 1
MSE_TD = np.zeros((N_walks, n_simulation))
MSE_TD = pd.DataFrame(MSE_TD)

for q in range(0, n_simulation):
    s = environment()
    state_value = np.zeros((7, 4))
    state_value = pd.DataFrame(state_value)
    state_value[0] = s.states
    for i in range(0, N_walks):
        action = s.select_action()
        current_state = s.current_state
        s.next_state(action)
        reward, is_terminate = s.check_terminate()
        state_value[1].iloc[current_state + 3] = state_value[1].iloc[current_state + 3] + 0.1 * (reward + gamma * state_value[1].iloc[s.current_state + 3] - state_value[1].iloc[current_state + 3])
        s.reset_game(is_terminate)
        MSE_TD[q].iloc[i] = ((state_value[1][1:-1] - actual_state_values) ** 2).sum()/5

MSE[1] = MSE_TD.mean(axis= 1)
MSE[2] = MSE_TD.mean(axis= 1)


import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

# Assuming 'MSE' is your DataFrame and it has columns '0' and '1'
plt.figure(figsize=(10, 6))

# Plotting the values of the two columns as lines
plt.plot(MSE.iloc[20:].index, MSE[0].iloc[20:], label='MC', color='blue')
plt.plot(MSE.iloc[20:].index, MSE[1].iloc[20:], label='TD (alpha = 0.05)', color='red')
plt.plot(MSE.iloc[20:].index, MSE[2].iloc[20:], label='TD (alpha = 0.1)', color='green')

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Averaged RMSE')
plt.title('Averaged RMSE of MC and TD Methods')

# Adding a legend to differentiate the lines
plt.legend()

plt.savefig('TD vs MC.png', dpi= 1200)
# Displaying the plot
plt.show()
