import encodings.idna

import numpy as np
import pandas as pd
import random

class Game:
    def __init__(self):
        self.actions = ['H', 'S']
        self.Cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    def RewardCalculator(self, playerscore, dealerscore):
        if playerscore > dealerscore:
            return 1
        elif playerscore < dealerscore:
            return -1
        else:
            return 0

    def CardGenerator(self):

        return random.choice(self.Cards)


class Environment:
    def __init__(self):
        self.playersumcard = 0
        self.dealerfirstcard = 0
        self.playerfirstcard = 0
        self.dealersumcard = 0
        self.gamma = 0.9
        self.statelist = []
        self.actionlist = []
        self.Qsum = np.zeros((22,11,2))
        self.Qcount = np.zeros((22,11,2))
        self.Q = np.zeros((22,11,2))

        self.game = Game()
        self.terminate_status = 0


    def playerplay(self, action):
        if action == 'H':
            card = self.game.CardGenerator()
            if self.playersumcard + card > 21:
                self.terminate_status = 1
                self.playersumcard += card
            else:
                self.playersumcard += card
        if action == 'S':
            self.terminate_status = 1

        return self.playersumcard, self.terminate_status

    def dealerplay(self, action):
        if action == 'H':
            card = self.game.CardGenerator()
            if self.dealersumcard + card > 21:
                self.terminate_status = 1
                self.dealersumcard += card
            else:
                self.dealersumcard += card
        if action == 'S':
            self.terminate_status = 1

        return self.dealersumcard, self.terminate_status

    def initialization(self):
        self.playerfirstcard = self.game.CardGenerator()
        self.playersumcard += self.playerfirstcard
        self.dealerfirstcard = self.game.CardGenerator()
        self.dealersumcard += self.dealerfirstcard

    def action_selector(self):
        if self.playersumcard <= 10:
            action = 'H'
        elif self.playersumcard < 20:
            rand = random.random()
            if rand < 0.5:
                action = 'H'
            else:
                action = 'S'
        else:
            action = 'S'

        return action

env = Environment()
for i in range (0, 5000):
    env.terminate_status = 0
    env.initialization()
    env.actionlist.append('H')
    env.statelist.append([env.playerfirstcard, env.dealerfirstcard])
    while env.terminate_status == 0:
        action = env.action_selector()
        env.playersumcard, env.terminate_status = env.playerplay(action)
        env.actionlist.append(action)
        env.statelist.append([env.playersumcard, env.dealerfirstcard])

    while env.dealersumcard < env.playersumcard:
        action = 'H'
        env.dealersumcard, env.terminate_status = env.dealerplay(action)

    if env.playersumcard > 21:
        reward = -1
    elif env.dealersumcard > 21:
        reward = 1
    else:
        reward = env.game.RewardCalculator(env.playersumcard, env.dealersumcard)

    G = 0
    env.actionlist = env.actionlist[1:]
    env.statelist = env.statelist[:-1]
    for o in range(len(env.actionlist)-1, -1, -1):
        l = len(env.actionlist)
        if o == l-1:
            G = reward
        else:
            G = env.gamma * G

        if env.actionlist[o] == 'S':
            ac = 1
        else:
            ac = 0
        env.Qsum[env.statelist[o][0]][env.statelist[o][1]][ac] += G
        env.Qcount[env.statelist[o][0]][env.statelist[o][1]][ac] += 1

    env.playersumcard = 0
    env.playerfirstcard = 0
    env.dealersumcard = 0
    env.dealerfirstcard = 0

Qsum = np.array(env.Qsum)
Qcount = np.array(env.Qcount)

Q = Qsum / Qcount

Strategy = np.zeros((22,11))
Strategy = pd.DataFrame(Strategy)

for i in range(0, 22):
    for j in range(0, 11):
        if Q[i][j][0] >= Q[i][j][1]:
            Strategy[j].iloc[i] = 1
        else:
            if i <= 10:
                Strategy[j].iloc[i] = 1
            else:
                Strategy[j].iloc[i] = 0

Strategy.drop([0], axis= 1, inplace= True)

import matplotlib.pyplot as plt

# Step 1: Create a sample DataFrame (replace this with your actual DataFrame)
df = Strategy

# Step 2: Create a figure and axis
fig, ax = plt.subplots()

# Step 3: Plot the DataFrame values
cax = ax.matshow(df, cmap='binary')

fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size
ax.set_aspect('auto')  # Adjust aspect ratio to make cells larger

# Step 3: Define a new color map (light green for 1 and light gray for 0)
colors = ['lightgray', 'lightgreen']
cmap = plt.cm.colors.ListedColormap(colors)

# Step 4: Plot the DataFrame values with the new color map
cax = ax.matshow(df, cmap=cmap)

# Step 5: Customize ticks, labels, and font
ax.set_xticks(np.arange(len(df.columns)))
ax.set_yticks(np.arange(len(df.index)))
ax.set_xticklabels(df.columns, fontname="Times New Roman", fontsize=14)
ax.set_yticklabels(df.index, fontname="Times New Roman", fontsize=14)

# Step 6: Add gridlines for clarity
ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)

# Step 7: Add a legend for the colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightgray', edgecolor='black', label='0'),
    Patch(facecolor='lightgreen', edgecolor='black', label='1')
]
ax.legend(handles=legend_elements, loc='upper right', title="Values")

# Step 8: Show the plot
plt.show()
