In case of changing environments, it so imperative to detect and act against changes. One idea to this end is using DynaQ+ algorithm in which the agent tries to see state-actions that has not seen for a while. In fact, by adding a term which is a function of last time of seeing a pair of state-action and consider it in action's selection criteria, agent can detect changes in the environment. This has been shown in the following picture which shows the performance of both methods for a maze in which locations of blocks change after 3000 steps. As it is clear, after changes in environment, DynaQ+ can detect this change more quickly and its rewards grows sharply compared with DynaQ.
![Changing Maze](https://github.com/user-attachments/assets/4c9e912d-1740-4244-98f8-13933ea5ada4)