import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import copy

class TwoHeadedDoubleDQN(nn.Module):
    def __init__(self, input_dim, output_dim, environment):
        super(TwoHeadedDoubleDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gameopsrl = environment

        # Online network
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.offensive_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.defensive_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        # Target network
        self.target_shared_network = copy.deepcopy(self.shared_network)
        self.target_offensive_head = copy.deepcopy(self.offensive_head)
        self.target_defensive_head = copy.deepcopy(self.defensive_head)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)

    def forward(self, x, target=False):
        if not target:
            shared_features = self.shared_network(x)
            offensive_q = self.offensive_head(shared_features)
            defensive_q = self.defensive_head(shared_features)
        else:
            shared_features = self.target_shared_network(x)
            offensive_q = self.target_offensive_head(shared_features)
            defensive_q = self.target_defensive_head(shared_features)
        return offensive_q, defensive_q

    def choose_action(self, state, epsilon, action_space, action_details):
        if np.random.random() > epsilon:
            # Use the neural network to get Q-values
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            offensive_q, defensive_q = self(state)
            
            # Calculate the ball difference and adjust defensive Q-values
            ball_difference = self.gameopsrl.game.current_player.score - self.gameopsrl.game.players[-1 * self.gameopsrl.game.current_player.color].score
            defensive_scale = 1.0 # + max(0, ball_difference - 1) * 0.1
            scaled_defensive_q = defensive_q * defensive_scale
            
            # Combine offensive and defensive Q-values
            combined_q = (offensive_q + scaled_defensive_q) / 2
            combined_q = combined_q.squeeze(0).detach().numpy()
            
            # Filter Q-values for valid actions
            valid_q_values = combined_q[action_space]
            
            # Print Q-values and corresponding actions for debugging
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

    def update(self, state, action_index, offensive_reward, defensive_reward, next_state, action_space, next_action_space, done):
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_index = torch.tensor(action_index, dtype=torch.long)  # Action index as scalar
        offensive_reward = torch.tensor([offensive_reward], dtype=torch.float32)
        defensive_reward = torch.tensor([defensive_reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        # Convert action_space and next_action_space to boolean masks
        action_mask = torch.zeros(self.output_dim, dtype=torch.bool)
        action_mask[action_space] = True
        next_action_mask = torch.zeros(self.output_dim, dtype=torch.bool)
        next_action_mask[next_action_space] = True

        # Expand the masks to match the shape of the Q-value tensors (batch size 1)
        action_mask = action_mask.unsqueeze(0)  # Shape [1, output_dim]
        next_action_mask = next_action_mask.unsqueeze(0)  # Shape [1, output_dim]

        # Get current Q-values for the state
        current_offensive_q, current_defensive_q = self(state)

        current_offensive_q = current_offensive_q.squeeze(0)

        current_defensive_q = current_defensive_q.squeeze(0)


        # Extract Q-values for the chosen action index
        current_offensive_q_value = current_offensive_q[action_index]
        current_defensive_q_value = current_defensive_q[action_index]

        # Compute target Q-values
        with torch.no_grad():
            # Use the online network to compute the next state Q-values
            next_offensive_q, next_defensive_q = self(next_state)
            
            # Apply the mask to set invalid actions to -inf
            next_offensive_q[~next_action_mask] = float('-inf')
            next_defensive_q[~next_action_mask] = float('-inf')

            # Choose the best actions for both heads
            best_offensive_actions = next_offensive_q.argmax(dim=1)
            best_defensive_actions = next_defensive_q.argmax(dim=1)

            # Use the target network to compute the target Q-values for the best actions
            next_offensive_q_target, next_defensive_q_target = self(next_state, target=True)
            max_next_offensive_q = next_offensive_q_target[0, best_offensive_actions]
            max_next_defensive_q = next_defensive_q_target[0, best_defensive_actions]

            # Calculate target Q-values using the Bellman equation
            target_offensive_q = offensive_reward + (1 - done) * 0.99 * max_next_offensive_q
            target_defensive_q = defensive_reward + (1 - done) * 0.99 * max_next_defensive_q

        # Compute the losses for both heads
        loss_offensive = F.mse_loss(current_offensive_q_value, target_offensive_q)
        loss_defensive = F.mse_loss(current_defensive_q_value, target_defensive_q)
        total_loss = loss_offensive + loss_defensive

        # Perform optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()




    def update_target_network(self, tau=0.001):
        """Soft update of the target network."""
        for target_param, online_param in zip(self.target_shared_network.parameters(), self.shared_network.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
        for target_param, online_param in zip(self.target_offensive_head.parameters(), self.offensive_head.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
        for target_param, online_param in zip(self.target_defensive_head.parameters(), self.defensive_head.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def action_to_index(self, action):
        

        start = action['start']
        end = action['end']
        action_type = action['type']
        
        index = (start[0][0] * 9 + start[0][1]) * 140 + (end[0][0] * 9 + end[0][1]) * 10 + action_type
        
        return index