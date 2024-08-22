from GameOpsRL import GameOpsRL
from Player import Player
class OutputAllPossibleActions():

    def __init__(self):
        player1 = Player("Black", 1)
        player2 = Player("White", -1)
        self.game = GameOpsRL(player1, player2)

    def output_all_possible_actions(self):
        # Initialize the dictionary to store all possible actions
        dict_possible_actions = {}
        action_index = 0  # Index counter for the actions

        # Track unique moves as tuples of (start, end, type)
        unique_moves = set()

        # Initialize the board by setting all cells to 0 (empty)
        self.clear_board()

        # Handle one ball moves and add to the dictionary
        action_index = self.add_actions_to_dict(dict_possible_actions, self.generate_actions_for_one_ball(), action_index, unique_moves)

        # Handle two ball moves using adjacent pairs and add to the dictionary
        action_index = self.add_actions_to_dict(dict_possible_actions, self.generate_actions_for_two_balls(), action_index, unique_moves)

        # Handle three ball moves using adjacent trios and add to the dictionary
        action_index = self.add_actions_to_dict(dict_possible_actions, self.generate_actions_for_three_balls(), action_index, unique_moves)

        # Finally, return the dictionary of possible actions
        return dict_possible_actions

    def clear_board(self):
        # Reset the board to all zeros (empty state)
        self.game.game.board.grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]

    def generate_actions_for_one_ball(self):
        possible_actions = []

        # Iterate over every cell on the board
        for row in range(len(self.game.game.board.grid)):
            for col in range(len(self.game.game.board.grid[row])):
                # Clear the board before placing the ball
                self.clear_board()

                # Place a single ball
                self.place_ball(row, col, self.game.game.current_player.color)

                # Get actions for one ball
                possible_actions.extend(self.get_actions())

        return possible_actions

    def generate_actions_for_two_balls(self):
        possible_actions = []

        # Get the adjacent pairs from the precomputed dictionary
        adjacent_pairs_dict = self.game.game.board.pairs_to_straight_lines

        # Iterate over every pair of adjacent cells
        for pair in adjacent_pairs_dict.keys():
            # Clear the board before placing the balls
            self.clear_board()

            # Place two adjacent balls
            self.place_ball(pair[0][0], pair[0][1], self.game.game.current_player.color)
            self.place_ball(pair[1][0], pair[1][1], self.game.game.current_player.color)

            # Get actions for two balls
            possible_actions.extend(self.get_actions())

        return possible_actions

    def generate_actions_for_three_balls(self):
        possible_actions = []

        # Get the adjacent trios from the precomputed dictionary
        adjacent_trios_dict = self.game.game.board.trios_to_straight_lines

        # Iterate over every trio of adjacent cells
        for trio in adjacent_trios_dict.keys():
            # Clear the board before placing the balls
            self.clear_board()

            # Place three adjacent balls
            self.place_ball(trio[0][0], trio[0][1], self.game.game.current_player.color)
            self.place_ball(trio[1][0], trio[1][1], self.game.game.current_player.color)
            self.place_ball(trio[2][0], trio[2][1], self.game.game.current_player.color)

            # Get actions for three balls
            possible_actions.extend(self.get_actions())

        return possible_actions

    def place_ball(self, row, col, color):
        # Place a ball of the current player's color at the specified position
        self.game.game.board.grid[row][col] = color

    def get_actions(self):
        # Get the available actions from the current game state
        action_space, action_details, _ = self.game.get_action_space()

        # Collect the unique actions
        unique_actions = []
        for action_idx in action_space:
            unique_actions.append(action_details[action_idx])

        return unique_actions

    def add_actions_to_dict(self, action_dict, actions, action_index, unique_moves):
        # Add actions to the dictionary, ensuring no duplicates
        for action in actions:
            # Create a tuple of (start, end, type) to check for uniqueness
            move_tuple = (tuple(action['start']), tuple(action['end']), action['type'])
            
            # If the move has not been added yet, add it to the dictionary and set
            if move_tuple not in unique_moves:
                unique_moves.add(move_tuple)
                action_dict[move_tuple] = action_index
                action_index += 1  # Increment the index for each unique action

        return action_index  # Return the updated index


outputter = OutputAllPossibleActions()
possible_actions = outputter.output_all_possible_actions()
print(possible_actions)