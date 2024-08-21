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

    def choose_action(self, state, epsilon, action_space, action_mask, action_details):
        valid_actions = [action for action, valid in zip(action_space, action_mask) if valid]
        full_action_mask = np.zeros(self.output_dim, dtype=bool)
        full_action_mask[:len(action_mask)] = action_mask

        if np.random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            offensive_q, defensive_q = self(state)
            ball_difference = self.gameopsrl.game.current_player.score - self.gameopsrl.game.players[-1* self.gameopsrl.game.current_player.color].score
            defensive_scale = 1.0 + max(0, ball_difference - 1) * 0.1
            scaled_defensive_q = defensive_q * defensive_scale
            combined_q = (offensive_q + scaled_defensive_q) / 2
            combined_q = combined_q.squeeze(0).detach().numpy()
            masked_q_values = np.where(full_action_mask, combined_q, float('-inf'))
            
            # Print Q-values and corresponding actions
            for i, (q_value, mask) in enumerate(zip(combined_q, full_action_mask)):
                if mask:
                    print(f"Action {i}: Q-value = {q_value}, Action = {action_details[i]}")

            action_index = np.argmax(masked_q_values)
            chosen_action_index = action_space[action_index]
        else:
            # Group actions by the number of balls involved
            one_ball_moves = []
            two_ball_moves = []
            three_ball_moves = []

            print('valid actions for random choosing:', valid_actions)
            for action_idx in valid_actions:
                action = action_details[action_idx]
                if isinstance(action['start'], tuple) and not isinstance(action['start'][0], tuple):
                    one_ball_moves.append(action_idx)
                elif len(action['start']) == 2:
                    two_ball_moves.append(action_idx)
                elif len(action['start']) == 3:
                    three_ball_moves.append(action_idx)

            print(f'One ball moves: {one_ball_moves}', f'Two ball moves: {two_ball_moves}', f'Three ball moves: {three_ball_moves}')
            # Define probabilities for selecting each group
            
            probs = [0.05, 0.35, 0.6]  # 10% for 1-ball, 30% for 2-ball, 60% for 3-ball moves

            # Normalize probabilities if some move types are not available
            available_moves = [one_ball_moves, two_ball_moves, three_ball_moves]
            available_probs = [prob for prob, moves in zip(probs, available_moves) if moves]
            if available_probs:
                normalized_probs = [p / sum(available_probs) for p in available_probs]
            else:
                # If no moves are available in any category, revert to uniform selection
                chosen_action_index = np.random.choice(valid_actions)
                action = action_details[chosen_action_index]
                return action, chosen_action_index

            # Choose a move type based on the probabilities
            chosen_move_type = np.random.choice([i for i, moves in enumerate(available_moves) if moves], p=normalized_probs)
            
            # Select a random action from the chosen move type
            if chosen_move_type == 0:
                chosen_action_index = np.random.choice(one_ball_moves)
            elif chosen_move_type == 1:
                chosen_action_index = np.random.choice(two_ball_moves)
            else:
                chosen_action_index = np.random.choice(three_ball_moves)

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
        done = torch.tensor([done], dtype=torch.float32)
        action_mask = torch.tensor(action_mask, dtype=torch.bool)
        next_action_mask = torch.tensor(next_action_mask, dtype=torch.bool)
        
        # Pad action masks if necessary
        if action_mask.shape[0] != self.output_dim:
            action_mask = F.pad(action_mask, (0, self.output_dim - action_mask.shape[0]), value=False)
        if next_action_mask.shape[0] != self.output_dim:
            next_action_mask = F.pad(next_action_mask, (0, self.output_dim - next_action_mask.shape[0]), value=False)
        
        action_mask = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)
        next_action_mask = torch.tensor(next_action_mask, dtype=torch.bool).unsqueeze(0)

        # Get current Q values
        current_offensive_q, current_defensive_q = self(state)
        current_offensive_q = current_offensive_q.squeeze(0)
        current_defensive_q = current_defensive_q.squeeze(0)


        # Compute target Q values
        with torch.no_grad():
            # Use online network for action selection
            next_offensive_q, next_defensive_q = self(next_state)
            next_offensive_q[~next_action_mask] = float('-inf')
            next_defensive_q[~next_action_mask] = float('-inf')
            best_offensive_actions = next_offensive_q.argmax(dim=1, keepdim=True)
            best_defensive_actions = next_defensive_q.argmax(dim=1, keepdim=True)

            # Use target network for value estimation
            next_offensive_q_target, next_defensive_q_target = self(next_state, target=True)
            max_next_offensive_q = next_offensive_q_target.gather(1, best_offensive_actions).squeeze(1)
            max_next_defensive_q = next_defensive_q_target.gather(1, best_defensive_actions).squeeze(1)

            target_offensive_q = offensive_reward + (1 - done) * 0.99 * max_next_offensive_q
            target_defensive_q = defensive_reward + (1 - done) * 0.99 * max_next_defensive_q

        # Compute loss
        loss_offensive = F.mse_loss(current_offensive_q, target_offensive_q)
        loss_defensive = F.mse_loss(current_defensive_q, target_defensive_q)
        total_loss = loss_offensive + loss_defensive

        # Optimize the model
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