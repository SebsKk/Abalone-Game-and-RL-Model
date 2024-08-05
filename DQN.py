import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from RewardSystem import RewardSystem
from GameOpsRL import GameOpsRL    
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, environment):
        super(DQN, self).__init__()
        # Define the neural network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.gameopsrl = environment
        self.input_dim = input_dim
        self.output_dim = output_dim 


        
    def forward(self, x):
        return self.network(x)

    def choose_action(self, state, epsilon, action_space, action_mask, action_details):
        # Choose an action based on epsilon-greedy policy
        valid_actions = [action for action, valid in zip(action_space, action_mask) if valid]

        full_action_mask = np.zeros(self.output_dim, dtype=bool)
        full_action_mask[:len(action_mask)] = action_mask

    
        if np.random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            q_values = self(state).squeeze(0)  # Remove batch dimension
            # Mask out invalid actions by setting their Q-values to a large negative value
            q_values = q_values.detach().numpy()  # Convert to numpy array for masking
            masked_q_values = np.where(full_action_mask, q_values, float('-inf'))
            action_index = np.argmax(masked_q_values)
            chosen_action_index = action_space[action_index]
        else:
            chosen_action_index = np.random.choice(valid_actions)  # Choose only among valid actions
        
        # Retrieve the detailed action using the chosen action index
        action = action_details[chosen_action_index]
        return action, chosen_action_index


    def transform_state_for_nn(self, state):
        grid, current_player = state
        one_hot_encoded_grid = []

        max_length = 9  # Maximum length based on the Abalone board specifications

        for row in grid:
            # Pad each row to the maximum length of 9
            padded_row = row + [-2] * (max_length - len(row))  # Pad with -2 which represents non-playable or non-existent cells
            for cell in padded_row:
                if cell == 1:
                    # Player 1
                    one_hot_encoded_grid.extend([1, 0, 0])
                elif cell == -1:
                    # Player 2
                    one_hot_encoded_grid.extend([0, 1, 0])
                elif cell == 0:
                    # Empty cell
                    one_hot_encoded_grid.extend([0, 0, 1])
                else:
                    # Non-existent cell due to padding
                    one_hot_encoded_grid.extend([0, 0, 0])

        # Convert to numpy array to ensure it can be processed by PyTorch
        return np.array(one_hot_encoded_grid, dtype=float)

    def update(self, state, action, reward, next_state, action_mask, next_action_mask, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        action_mask = torch.tensor(action_mask, dtype=torch.bool)
        next_action_mask = torch.tensor(next_action_mask, dtype=torch.bool)

        # Ensure action_mask and next_action_mask have the correct shape
        if action_mask.shape[0] != self.output_dim:
            action_mask = F.pad(action_mask, (0, self.output_dim - action_mask.shape[0]), value=False)
        if next_action_mask.shape[0] != self.output_dim:
            next_action_mask = F.pad(next_action_mask, (0, self.output_dim - next_action_mask.shape[0]), value=False)

        current_q_values = self(state).squeeze(0)
        current_q_value = current_q_values[action]

        with torch.no_grad():
            next_q_values = self(next_state).squeeze(0)
            next_q_values[~next_action_mask] = float('-inf')
            max_next_q = next_q_values.max()
            target_q_value = reward + (1 - done) * 0.99 * max_next_q

        loss = nn.functional.mse_loss(current_q_value.unsqueeze(0), target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def action_to_index(self, action):
        

        start = action['start']
        end = action['end']
        action_type = action['type']
        
        # Calculate index based on start, end, and type
        index = (start[0][0] * 9 + start[0][1]) * 140 + (end[0][0] * 9 + end[0][1]) * 10 + action_type
        
        return index


