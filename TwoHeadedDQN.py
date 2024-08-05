import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from RewardSystem import RewardSystem
from GameOpsRL import GameOpsRL    
import torch.nn.functional as F

class TwoHeadedDQN(nn.Module):
    def __init__(self, input_dim, output_dim, environment):
        super(TwoHeadedDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gameopsrl = environment

        # Shared layers
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Offensive head
        self.offensive_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        # Defensive head
        self.defensive_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    
    def forward(self, x):
        shared_features = self.shared_network(x)
        offensive_q = self.offensive_head(shared_features)
        defensive_q = self.defensive_head(shared_features)
        return offensive_q, defensive_q

    def choose_action(self, state, epsilon, action_space, action_mask, action_details):
        valid_actions = [action for action, valid in zip(action_space, action_mask) if valid]
        full_action_mask = np.zeros(self.output_dim, dtype=bool)
        full_action_mask[:len(action_mask)] = action_mask
        

        if np.random.random() > epsilon:
            
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            offensive_q, defensive_q = self(state)
            ball_difference = self.gameopsrl.game.current_player.score - self.gameopsrl.game.players[-1* self.gameopsrl.game.current_player.color].score
            defensive_scale = 1.0 + max(0, ball_difference - 1) * 0.5
            scaled_defensive_q = defensive_q * defensive_scale
            combined_q = (offensive_q + defensive_q) / 2
            combined_q = combined_q.squeeze(0).detach().numpy()
            masked_q_values = np.where(full_action_mask, combined_q, float('-inf'))
            action_index = np.argmax(masked_q_values)
            chosen_action_index = action_space[action_index]
        else:
            chosen_action_index = np.random.choice(valid_actions)

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
    
    def update(self, state, action, offensive_reward, defensive_reward, next_state, action_mask, next_action_mask, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        offensive_reward = torch.tensor([offensive_reward], dtype=torch.float32)
        defensive_reward = torch.tensor([defensive_reward], dtype=torch.float32)
        action_mask = torch.tensor(action_mask, dtype=torch.bool)
        next_action_mask = torch.tensor(next_action_mask, dtype=torch.bool)

        if action_mask.shape[0] != self.output_dim:
            action_mask = F.pad(action_mask, (0, self.output_dim - action_mask.shape[0]), value=False)
        if next_action_mask.shape[0] != self.output_dim:
            next_action_mask = F.pad(next_action_mask, (0, self.output_dim - next_action_mask.shape[0]), value=False)

        current_offensive_q, current_defensive_q = self(state)
        current_offensive_q = current_offensive_q.squeeze(0)
        current_defensive_q = current_defensive_q.squeeze(0)

        with torch.no_grad():
            next_offensive_q, next_defensive_q = self(next_state)
            next_offensive_q = next_offensive_q.squeeze(0)
            next_defensive_q = next_defensive_q.squeeze(0)
            next_offensive_q[~next_action_mask] = float('-inf')
            next_defensive_q[~next_action_mask] = float('-inf')
            max_next_offensive_q = next_offensive_q.max()
            max_next_defensive_q = next_defensive_q.max()

        target_offensive_q = offensive_reward + (1 - done) * 0.99 * max_next_offensive_q
        target_defensive_q = defensive_reward + (1 - done) * 0.99 * max_next_defensive_q

        loss_offensive = nn.functional.mse_loss(current_offensive_q[action].unsqueeze(0), target_offensive_q)
        loss_defensive = nn.functional.mse_loss(current_defensive_q[action].unsqueeze(0), target_defensive_q)
        total_loss = loss_offensive + loss_defensive

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


    def action_to_index(self, action):
        

        start = action['start']
        end = action['end']
        action_type = action['type']
        
        index = (start[0][0] * 9 + start[0][1]) * 140 + (end[0][0] * 9 + end[0][1]) * 10 + action_type
        
        return index