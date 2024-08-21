
from MCTSNode import MCTSNode
import torch
import copy
import numpy as np
from numpy import ndarray

class MCTS:
    def __init__(self, neural_network, environment, c_puct=1.0, num_simulations=200, max_depth=50):
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.environment = environment

    def run(self, state, non_transformed_state):
        # Transform the original state for NN input
        root = MCTSNode(state, self.environment)
        
        print(f"Starting MCTS with {self.num_simulations} simulations")
        print(f'State that we got in MCTS: {state}')

        for sim in range(self.num_simulations):
            print(f"Simulation {sim + 1}/{self.num_simulations}")
            node = root


            # Create a deep copy of the game state for simulation
            simulation_state = copy.deepcopy(state)

            # Simulate from the root node with depth tracking
            node, depth = self.simulate_one_path(node, simulation_state, non_transformed_state, depth=0)

            # Backpropagate the result
            value = self.simulate(node.state)
            node.backpropagate(value)
            print(f"Backpropagated value: {value} to depth {depth}")

        # Choose the action with the most visits
        best_action = max(root.children.items(), key=lambda child: child[1].visit_count)[0]
        print(f"Best action selected after MCTS: {best_action}")
        return best_action
    
    def restore_state(self, state):
        self.environment.game.board.grid = state[0]
        self.environment.game.current_player = self.environment.game.players[0] if state[1][0] == self.environment.game.players[0].color else self.environment.game.players[1]


    def simulate_one_path(self, node, simulation_state, non_transformed_state, depth):
        print(f"Simulating one path. Starting at depth {depth}")

        if node is None:
            print(f"Error: Node is None at depth {depth}. Exiting simulation.")
            return None, depth
        
        original_state = copy.deepcopy(non_transformed_state)

        # Check if maximum depth is reached
        if depth >= self.max_depth:
            print(f"Reached maximum depth {self.max_depth}")
            return node, depth

        while depth < self.max_depth:
            # If the node is not fully expanded, expand it
            if not node.is_fully_expanded():
                print(f"Expanding node at depth {depth}")
                action_space, valid_actions, action_mask = self.environment.get_action_space()

                if not isinstance(simulation_state, ndarray):
                    simulation_state = self.neural_network.transform_state_for_nn(simulation_state)

                simulation_state = torch.tensor(simulation_state).unsqueeze(0).reshape(1, 3, 9, 9).float()
                action_probs, value = self.neural_network.predict(simulation_state)

                valid_action_indices = np.where(action_mask)[0]
                valid_probs = action_probs[valid_action_indices]
                # Add some noise to encourage exploration
                noise = np.random.dirichlet([0.3] * len(valid_probs))
                valid_probs = 0.75 * valid_probs + 0.25 * noise
                
                # Use temperature to control exploration
                temperature = 1.0  # Adjust this value; higher for more exploration
                valid_probs = np.power(valid_probs, 1/temperature)
                valid_probs /= np.sum(valid_probs)
                
                best_action_index = np.random.choice(valid_action_indices, p=valid_probs)
                selected_action = valid_actions[best_action_index]

                print(f"Selected action details for expansion: {selected_action}")
                next_state, _, done = self.environment.step(selected_action)
                print(f'Next state after expansion: {next_state} from action {selected_action}')

                transformed_next_state = self.neural_network.transform_state_for_nn(next_state)

                node = node.expand(best_action_index, transformed_next_state, action_probs[best_action_index])

                simulation_state = next_state
                depth += 1

                if done:
                    print(f"Game ended at depth {depth}")
                    break
            else:
                # If the node is fully expanded, select a child node
                action_space, valid_actions, action_mask = self.environment.get_action_space()
                action_index, node = node.select(self.c_puct, valid_actions)

                if action_index is None or node is None:
                    print(f"No valid child nodes found at depth {depth}")
                    break

                action_details = valid_actions[action_index]
                print(f'Applying action {action_details} to the simulation state')
                next_state, _, done = self.environment.step(action_details)

                print(f'Next state after selection: {next_state} from action {action_details}')
                simulation_state = next_state
                depth += 1

                if done:
                    print(f"Game ended at depth {depth}")
                    break

        # At the end, restore the original state from before the simulation was played out
        self.restore_state(original_state)

        return node, depth
    def simulate(self, state):
        # Ensure the state has the correct shape for the CNN
        # Assuming your board size is 9x9 and you have 3 channels
        print(f"Simulating from state")
        
        # Ensure the state is a tensor and reshape it appropriately for the CNN
        state_tensor = torch.tensor(state).unsqueeze(0).reshape(1, 3, 9, 9).float()
        
        # Pass the reshaped state to the neural network to predict the value
        _, value = self.neural_network.predict(state_tensor)
        
        print(f"Predicted value: {value}")
        return value