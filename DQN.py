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

    def choose_action(self, state, epsilon, action_space, action_details):
        

    
        if np.random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self(state).squeeze(0).detach().numpy()
            valid_q_values = q_values[action_space]

            for i, action_idx in enumerate(action_space):
                print(f"Action {action_idx}: Q-value = {valid_q_values[i]}, Action = {action_details[action_idx]}")
            
            # Choose the action with the highest Q-value
            chosen_action_index = action_space[np.argmax(valid_q_values)]
        else:
            # Group actions by the number of balls involved
            one_ball_moves = []
            two_ball_moves = []
            three_ball_moves = []
            
            for action_idx in action_space:
                action = action_details[action_idx]
                if isinstance(action['start'], tuple) and not isinstance(action['start'][0], tuple):
                    one_ball_moves.append(action_idx)
                elif len(action['start']) == 2:
                    two_ball_moves.append(action_idx)
                elif len(action['start']) == 3:
                    three_ball_moves.append(action_idx)
            

            
            # Define and normalize probabilities for selecting each group
            probs = [0.15, 0.35, 0.5]  
            available_moves = [one_ball_moves, two_ball_moves, three_ball_moves]
            available_probs = [prob for prob, moves in zip(probs, available_moves) if moves]
            
            if available_probs:
                normalized_probs = [p / sum(available_probs) for p in available_probs]
                chosen_move_type = np.random.choice([i for i, moves in enumerate(available_moves) if moves], p=normalized_probs)
                
                if chosen_move_type == 0:
                    chosen_action_index = np.random.choice(one_ball_moves)
                elif chosen_move_type == 1:
                    chosen_action_index = np.random.choice(two_ball_moves)
                else:
                    chosen_action_index = np.random.choice(three_ball_moves)
            else:
                # If no moves are available in any category, revert to uniform selection
                chosen_action_index = np.random.choice(action_space)
        
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

    def update(self, state, action_index, reward, next_state, action_space, next_action_space, done):
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_index = torch.tensor(action_index, dtype=torch.long)  # Action index as scalar
        reward = torch.tensor([reward], dtype=torch.float32)
        
        done = torch.tensor([done], dtype=torch.float32)

        # Convert action_space and next_action_space to boolean masks
        action_mask = torch.zeros(self.output_dim, dtype=torch.bool)
        action_mask[action_space] = True
        next_action_mask = torch.zeros(self.output_dim, dtype=torch.bool)
        next_action_mask[next_action_space] = True

        # Expand the masks to match the shape of the Q-value tensors (batch size 1)
        action_mask = action_mask.unsqueeze(0)  # Shape [1, output_dim]
        next_action_mask = next_action_mask.unsqueeze(0)  # Shape [1, output_dim]

    

        current_q_values = self(state).squeeze(0)
        current_q_value = current_q_values[action_index]

        with torch.no_grad():
            next_q_values = self(next_state)
            next_q_values[~next_action_mask] = float('-inf')
            max_next_q = next_q_values.argmax(dim=1)
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


