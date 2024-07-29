import unittest
from QLearningTable import QLearningTable

class TestQLearningTable(unittest.TestCase):

    def setUp(self):
        self.actions = [0, 1, 2, 3]  # Assuming these represent some kind of action indices
        self.ql = QLearningTable(self.actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1)


    def test_learning(self):
        initial_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,-1,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], -1)  
        action = 3 # Move one  ball to the left
        reward = 1  # Let's say the agent gets a reward of 1 for moving there
        next_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,-1,0,0,0,0,0],
            [0,0,0,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], 1)  

        self.ql.learn(initial_state, action, reward, next_state)

        # Check if the Q-value for the initial state and the chosen action has been updated
        initial_state_key = self.ql.state_to_key(initial_state)
        self.assertNotEqual(self.ql.q_table[initial_state_key][action], 0)

    def test_action_selection(self):
        # Example state aligned with the actual game's complexity
        state = ([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]
        ], -1)  

        # Choose an action using the QLearningTable
        action_index = self.ql.choose_action(state)

        
        self.assertIn(action_index, range(len(self.actions)))  # Checks if the returned index is valid

        # Additionally, we can verify that the returned action is one of the possible actions
        
        chosen_action = self.actions[action_index]
        self.assertIn(chosen_action, self.actions)

if __name__ == '__main__':
    unittest.main()