import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNPolicyValueNetwork(nn.Module):
    def __init__(self, board_size, num_actions):
        super(CNNPolicyValueNetwork, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions

        # Convolutional layers for shared feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, num_actions)
        
        # Value head
        self.value_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Shared layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Policy head
        policy_x = F.relu(self.policy_conv(x))
        policy_x = policy_x.view(policy_x.size(0), -1)  # Flatten the tensor
        policy_logits = self.policy_fc(policy_x)

        # Value head
        value_x = F.relu(self.value_conv(x))
        value_x = value_x.view(value_x.size(0), -1)  # Flatten the tensor
        value_x = F.relu(self.value_fc1(value_x))
        state_value = torch.tanh(self.value_fc2(value_x))  # Output range is [-1, 1]

        return policy_logits, state_value

    def predict(self, state):
        # Expect state to be preprocessed to the shape [batch_size, channels, height, width]
        with torch.no_grad():
            policy_logits, state_value = self(state)
            policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).detach().numpy()
            state_value = state_value.item()
        return policy_probs, state_value
    
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