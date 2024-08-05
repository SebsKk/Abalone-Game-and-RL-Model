import unittest
from unittest.mock import Mock, patch
import torch
import numpy as np
from DQN import DQN
from Player import Player
from GameOpsRL import GameOpsRL

class TestDQN(unittest.TestCase):
    def setUp(self):
        self.input_dim = 243
        self.output_dim = 140
        self.mock_environment = Mock(spec=GameOpsRL)
        self.dqn = DQN(self.input_dim, self.output_dim, self.mock_environment)

    def test_forward_pass_output_shape(self):
        sample_input = torch.rand((1, self.input_dim))
        output = self.dqn(sample_input)
        self.assertEqual(output.shape, (1, self.output_dim))

    def test_transform_state_for_nn(self):
        grid = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ]
        current_player = 1
        transformed_state = self.dqn.transform_state_for_nn((grid, current_player))
        self.assertEqual(len(transformed_state), self.input_dim)

    @patch('numpy.random.random')
    def test_choose_action_greedy(self, mock_random):
        mock_random.return_value = 1  # Ensure greedy choice
        state = np.random.rand(self.input_dim)
        action_space = list(range(self.output_dim))
        action_mask = [True] * self.output_dim
        action_details = {i: f'Action {i}' for i in range(self.output_dim)}
        
        with patch.object(self.dqn, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([range(self.output_dim)], dtype=torch.float32)
            action = self.dqn.choose_action(state, 0, action_space, action_mask, action_details)
        
        self.assertEqual(action, 'Action 139')  # Highest Q-value action

    @patch('numpy.random.random')
    @patch('numpy.random.choice')
    def test_choose_action_random(self, mock_choice, mock_random):
        mock_random.return_value = 0  # Ensure random choice
        mock_choice.return_value = 42
        state = np.random.rand(self.input_dim)
        action_space = list(range(self.output_dim))
        action_mask = [True] * self.output_dim
        action_details = {i: f'Action {i}' for i in range(self.output_dim)}
        
        action = self.dqn.choose_action(state, 1, action_space, action_mask, action_details)
        
        self.assertEqual(action, 'Action 42')

    def test_update_changes_parameters(self):
        state = np.random.rand(self.input_dim)
        next_state = np.random.rand(self.input_dim)
        action = 0
        reward = 1
        done = False

        old_params = [param.clone().detach() for param in self.dqn.parameters()]
        self.dqn.update(state, action, reward, next_state, done)
        new_params = [param.clone().detach() for param in self.dqn.parameters()]

        self.assertFalse(all(torch.equal(old, new) for old, new in zip(old_params, new_params)))

if __name__ == '__main__':
    unittest.main()