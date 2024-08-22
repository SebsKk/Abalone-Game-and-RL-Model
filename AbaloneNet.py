import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import math
import copy
import random 
import json

class AbaloneNet(nn.Module):
    def __init__(self, input_dim, output_dim, environment):
        super(AbaloneNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gameopsrl = environment

        # Reshape input to 2D for convolutional layers
        self.board_size = 9  
        self.input_channels = 3 

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(256 * self.board_size * self.board_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)
        )

        # Value head (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)

    def forward(self, x):
        # Reshape input to 2D
        x = x.view(-1, self.input_channels, self.board_size, self.board_size)
        
        conv_out = self.conv_layers(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        shared_out = self.shared_fc(flattened)
        
        policy = self.policy_head(shared_out)
        value = self.value_head(shared_out)
        
        return policy, value

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

    def predict(self, state):
        state_tensor = torch.tensor(self.transform_state_for_nn(state), dtype=torch.float32).unsqueeze(0)
        policy, value = self(state_tensor)
        return policy.squeeze(0).detach().numpy(), value.item()

    def train(self, states, mcts_probs, winners):
        states_tensor = torch.tensor([self.transform_state_for_nn(state) for state in states], dtype=torch.float32)
        
        mcts_probs_tensor = torch.tensor(mcts_probs, dtype=torch.float32)
        
        winners_tensor = torch.tensor(winners, dtype=torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        policy, value = self(states_tensor)

        policy_loss = -torch.mean(torch.sum(mcts_probs_tensor * torch.log(policy + 1e-8), dim=1))
        value_loss = F.mse_loss(value, winners_tensor)
        loss = policy_loss + value_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

class MCTSNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p

    def expand(self, action_probs, action_details):
        for action, prob in zip(action_details, action_probs):
            action_tuple = (action['start'], action['end'], action['type'])
            if action_tuple not in self.children and prob > 0:

                #print(f'expanding action: {action_tuple}')
                self.children[action_tuple] = MCTSNode(self, prob)

    @staticmethod
    def _dict_to_tuple(action):
        return (action['start'], action['end'], action['type'])

    def select(self, c_puct):
        return max(self.children.items(), 
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        u = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + u

    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, policy_value_fn, game_ops, c_puct=5, n_playout=300, initial_epsilon=0.9, epsilon_decay=0.995):
        self.root = MCTSNode(None, 1.0)
        self.policy = policy_value_fn
        self.game_ops = game_ops
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.1
        self.total_playouts = 0

    def _select_action(self, node, action_space, action_details):
        if random.random() < self.epsilon:
            action = random.choice(list(action_details.values()))
            return action, node

        while not node.is_leaf():
            action, child_node = node.select(self.c_puct)
            
            # If action is already a dictionary, return it
            if isinstance(action, dict):
                #print(f'Action selected: {action}, Node visits: {child_node.n_visits}')
                return action, child_node
            
            # If action is a tuple, convert it to a dictionary
            if isinstance(action, tuple):
                for idx, action_dict in action_details.items():
                    if (action[0] == action_dict['start'] and 
                        action[1] == action_dict['end'] and 
                        action[2] == action_dict['type']):
                        #print(f'Action selected: {action}, Node visits: {child_node.n_visits}')
                        return action_dict, child_node
            
            # If we couldn't find a matching action, move to the child node and continue
            node = child_node

        # If we've reached a leaf node, return a random action
        return random.choice(list(action_details.values())), node

    def playout(self, state):
        node = self.root
        path = []
        depth = 0
        max_depth = 50  # Set a maximum depth to prevent infinite loops

        while depth < max_depth:
            if node.is_leaf():
                #print(f'Leaf node reached at depth {depth}. Visit count: {node.n_visits}')
                break

            action_space, action_details, _ = self.game_ops.get_action_space()
            action_dict, node = self._select_action(node, action_space, action_details)
            
            action_tuple = (action_dict['start'], action_dict['end'], action_dict['type'])
            
            path.append((action_tuple, node))
            state, reward, done = self.game_ops.step(action_dict)
            
            
            depth += 1
            
            if done:
                break


        action_probs, leaf_value = self.policy(state)

        #print(f'Action probs: {action_probs}, Leaf value: {leaf_value}')
        if not self.game_ops.is_game_over():
            action_space, action_details, _ = self.game_ops.get_action_space()
            action_details_list = list(action_details.values()) 
            node.expand(action_probs, action_details_list)
            #print(f"Expanded leaf node with {len(node.children)} children")

        self.backpropagate(path, leaf_value, self.game_ops.game.current_player.color)

        #print('After backpropagation:')
        #for action_tuple, node in path:
            #print(f'Action: {action_tuple}, Node visits: {node.n_visits}')

        return path, leaf_value
    
    def backpropagate(self, path, leaf_value, to_play):
        # Debugging: Backpropagation started
        #print("Backpropagation started:")
        
        for action, node in reversed(path):
            # Update node visits and Q value
            node.n_visits += 1
            
            # Update Q value as a running average
            node.Q += (leaf_value - node.Q) / node.n_visits
            
            # Debugging: Print the current node's stats
            #print(f'Action: {action}, Node visits: {node.n_visits}, Q value: {node.Q}')
            
            # Reverse the leaf value for alternating players
            leaf_value = -leaf_value
        
        # Debugging: Backpropagation completed
        #print("Backpropagation completed.")


    def make_move(self, selected_move):
            """Apply the selected move to the game and update the tree root."""
            self.game_ops.step(selected_move)
            self.update_with_move(selected_move)


    def collect_visits(self, node, action_details, visit_dict=None):
        if visit_dict is None:
            visit_dict = {}

        for act, child_node in node.children.items():
            # Find the corresponding action details
            for idx, action in action_details.items():
                if (act[0] == action['start'] and act[1] == action['end'] and act[2] == action['type']):
                    # Store the visit count in the dictionary
                    move_key = (action['start'], action['end'], action['type'])
                    visit_dict[move_key] = child_node.n_visits
                    # Recursively collect visits from the child nodes
                    self.collect_visits(child_node, action_details, visit_dict)
                    break

        return visit_dict
    
    def get_move_probs(self, state, temp=1e-2):
        #print(f'State that is being passed to get_move_probs: {state}')  
        
        # Deep copy the initial state and player scores/colors
        original_state = copy.deepcopy(state)
        original_scores = {i: player.score for i, player in enumerate(self.game_ops.game.players)}
        original_colors = {i: player.color for i, player in enumerate(self.game_ops.game.players)}
        original_player = self.game_ops.game.current_player
        

        for _ in range(self.n_playout):
            
            # Restore the original state and player
            self.game_ops.game.board.grid = copy.deepcopy(original_state[0])
            self.game_ops.game.current_player = original_player
            

            
            for i, player in enumerate(self.game_ops.game.players):
                player.score = original_scores[i]
                player.color = original_colors[i]

            # Perform a playout from the current state
            self.playout(copy.deepcopy(original_state))
            #print('Playout done')
            self.total_playouts += 1
            #print(f'Total playouts: {self.total_playouts}')

        #print('node visits: ', self.root.n_visits)


        # Restore the original state and player after all playouts
        self.game_ops.game.board.grid = original_state[0]
        self.game_ops.game.current_player = original_player


        
        for i, player in enumerate(self.game_ops.game.players):
            player.score = original_scores[i]
            player.color = original_colors[i] # Restore each player's color after playouts

        # Decay epsilon (if applicable in your algorithm)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        #print(f'current game state after restoring is {self.game_ops.get_current_state()}')
        # Get the action space and details to match the move probabilities
        action_space, action_details, _ = self.game_ops.get_action_space()
        
        #visit_dict = self.collect_visits(self.root, action_details=action_details)

        #print("Move visits: ", visit_dict)

        def compare_actions(child_key, action_detail):
            return (child_key[0] == action_detail['start'] and 
                    child_key[1] == action_detail['end'] and 
                    child_key[2] == action_detail['type'])


        # Get visit counts for each action from the MCTS tree
        act_visits = []
        print(f'action details to be checked: {action_details}')
        for act, node in self.root.children.items():
            print(f"Root child: {act}, Visits: {node.n_visits}")
            for idx, action in action_details.items():
                if compare_actions(act, action):
                    print(f"Matched action: {action}, Visits: {node.n_visits}")
                    act_visits.append((idx, node.n_visits))
        #print(f'act_visits: {act_visits}')
        if not act_visits:
            #print('No valid actions found in MCTS tree. Returning uniform distribution.')
            return [action_details[act] for act in action_space], np.ones(len(action_space)) / len(action_space)

        acts, visits = zip(*act_visits)
        visits = np.array(visits, dtype=np.float64)

        if temp <= 1e-8:  # If temperature is virtually zero, use argmax
            best_act = acts[np.argmax(visits)]
            probs = np.zeros(len(acts))
            probs[list(acts).index(best_act)] = 1.0
        else:
            # Apply softmax with temperature
            logits = np.log(visits + 1e-10)  # Add small constant to avoid log(0)
            logits = logits / temp
            logits = logits - np.max(logits)  # Subtract max for numerical stability
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits)

        # Map actions to probabilities
        acts_prob_dict = dict(zip(acts, probs))

        #print(f'Acts prob dict: {acts_prob_dict}')  # Debug

        # Prepare final action and probability lists
        final_acts = []
        final_probs = []
        for act in action_space:
            if act in acts_prob_dict:
                final_acts.append(action_details[act])
                final_probs.append(acts_prob_dict[act])
            else:
                # Assign small probability to unexplored actions
                final_acts.append(action_details[act])
                final_probs.append(1e-8)

        

        return final_acts, final_probs
        
    @staticmethod
    def _softmax(x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs
        
    def update_with_move(self, last_move):
        last_move_key = (last_move['start'], last_move['end'], last_move['type'])
        if last_move_key in self.root.children:
            self.root = self.root.children[last_move_key]
            self.root.parent = None
        else:
            self.root = MCTSNode(None, 1.0)

        action_space, action_details, _ = self.game_ops.get_action_space()
        action_probs, _ = self.policy(self.game_ops.get_current_state())
        self.root.expand(action_probs, list(action_details.values()))