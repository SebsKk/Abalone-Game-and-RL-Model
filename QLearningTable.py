from GameOpsRL import GameOpsRL
import numpy as np



class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = {}

    def state_to_key(self, state):
        """Convert a state given as (grid, current_player) to a hashable key."""
        grid, current_player = state
        flat_grid = tuple(item for sublist in grid for item in sublist)  # Flatten the grid to a tuple
        return str(flat_grid + tuple([current_player]))  # Wrap current_player in a list before converting to tuple


    def choose_action(self, observation):
        # Convert observation to a key that can be used in a dictionary
        observation_key = self.state_to_key(observation)
        self.check_state_exist(observation_key)

        if np.random.uniform() > self.epsilon:
            state_action = self.q_table[observation_key]
            action_index = np.argmax(state_action)
        else:
            action_index = np.random.choice(len(self.actions))

        return action_index

    def learn(self, s, a, r, s_):
        s_key = self.state_to_key(s)
        s_key_ = self.state_to_key(s_)

        self.check_state_exist(s_key_)
        self.check_state_exist(s_key)

        q_predict = self.q_table[s_key][a]
        q_target = r + self.gamma * np.max(self.q_table[s_key_])

        self.q_table[s_key][a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table:
            # append new state to q table
            self.q_table[state] = [0 for _ in range(len(self.actions))]
