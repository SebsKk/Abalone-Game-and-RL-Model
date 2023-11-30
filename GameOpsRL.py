from GameRL import GameRL
from Player import Player
from Board import Board 
from gym import spaces
from collections import defaultdict

class GameOpsRL:
    def __init__(self, player1, player2):
        self.game = GameRL(player1, player2, Board())
        self.current_player = player1
        self.reward_structure = {
            "win": 100,
            "lose": -100,
            "invalid_move": -5,
            "valid_move": 1
        }
        self.observation_space = spaces.Box(low=-1, high=1, shape=(len(self.game.board.grid), len(max(self.game.board.grid, key=len))), dtype=int)
        self.adjacent_cells_dict = self.all_adjacent_balls(self.game.board)


    def reset(self):
        self.game.initialize_game()
        return self.get_current_state()

    def step(self, action):
        balls_start, balls_end = action
        success = False
        if len(balls_end) == len(balls_start):
            balls_start, balls_end = self.sort_balls(balls_start, balls_end)
            success = self.make_move(balls_start, balls_end)
        else:
            # If balls_start and balls_end have different lengths, treat as an invalid move
            success = False
        
        # Define rewards
        reward = self.reward_structure["valid_move"] if success else self.reward_structure["invalid_move"]
        
        # Check if the game is over
        done = self.is_game_over()
        if done:
            winner = self.get_winner()
            if winner == self.current_player:
                reward = self.reward_structure["win"]
            else:
                reward = self.reward_structure["lose"]

        return self.get_current_state(), reward, done
    
    def find_player_balls(self, board, player):
 
        player_positions = []
        for row_index, row in enumerate(board):
            for col_index, cell in enumerate(row):
                if cell == player:
                    player_positions.append((row_index, col_index))
        return player_positions

    def all_adjacent_balls(self, board):
    # build a dictionary of rows to number of columns
        rows_size = {0:4, 1:5, 2:6, 3:7, 4:8, 5:7, 6:6, 7:5, 8:4}
        all_adjacent_cells = {}

        for index, row in enumerate(board):
            for col_n, value in enumerate(row):
                adjacent_cells = []
                
                # Handling the rows < 4
                if index < 4:
                    if col_n == 0 and index != 0:
                        adjacent_cells.extend([(index, col_n + 1), (index + 1, col_n), (index - 1, col_n), (index + 1, col_n + 1)])
                    elif col_n == 0 and index == 0:
                        adjacent_cells.extend([(index, col_n + 1), (index + 1, col_n), (index + 1, col_n + 1)])     
                    elif col_n == rows_size[index] and index != 0:
                        adjacent_cells.extend([(index, col_n - 1), (index + 1, col_n), (index - 1, col_n - 1), (index + 1, col_n - 1)])
                    elif col_n == 4 and index == 0:
                        adjacent_cells.extend([(index, col_n - 1), (index + 1, col_n), (index + 1, col_n - 1)])
                    elif index == 0:
                        adjacent_cells.extend([(index, col_n + 1), (index, col_n - 1), (index + 1, col_n), (index + 1, col_n + 1)])    
                    else:
                        adjacent_cells.extend([(index, col_n + 1), (index, col_n - 1), (index + 1, col_n), (index + 1, col_n + 1), (index - 1, col_n), (index - 1, col_n - 1)])

                # Handling the rows > 4
                if index > 4:
                    if col_n == 0 and index != 8:
                        adjacent_cells.extend([(index, col_n + 1), (index + 1, col_n), (index - 1, col_n), (index - 1, col_n + 1)])
                    if col_n == 0 and index == 8:
                        adjacent_cells.extend([(index, col_n + 1), (index - 1, col_n + 1), (index - 1, col_n)])  
                    elif col_n == rows_size[index] and index != 8:
                        adjacent_cells.extend([(index, col_n - 1), (index + 1, col_n - 1), (index - 1, col_n), (index - 1, col_n + 1)])
                    elif col_n == rows_size[index] and index == 8:
                        adjacent_cells.extend([(index, col_n - 1), (index - 1, col_n), (index - 1, col_n + 1)])
                    elif index == 8:
                        adjacent_cells.extend([(index, col_n + 1), (index, col_n - 1), (index - 1, col_n), (index - 1, col_n + 1)]) 
                    else:
                        adjacent_cells.extend([(index, col_n + 1), (index, col_n - 1), (index + 1, col_n), (index + 1, col_n - 1), (index - 1, col_n), (index - 1, col_n + 1)])

                # Handling the middle row (index == 4)
                if index == 4:
                    if col_n == 0:
                        adjacent_cells.extend([(index, col_n + 1), (index + 1, col_n), (index - 1, col_n)])  
                    elif col_n == rows_size[index]:
                        adjacent_cells.extend([(index, col_n - 1), (index + 1, col_n - 1), (index - 1, col_n - 1)])
                    else:
                        adjacent_cells.extend([(index, col_n + 1), (index, col_n - 1), (index + 1, col_n), (index + 1, col_n - 1), (index - 1, col_n), (index - 1, col_n - 1)])

                # Filter out invalid adjacent cells
                valid_adjacent_cells = [(i, j) for i, j in adjacent_cells if 0 <= i < len(board) and 0 <= j < len(board[i])]

                # Add the valid adjacent cells to the dictionary
                all_adjacent_cells[(index, col_n)] = valid_adjacent_cells

        return all_adjacent_cells
    
    def parallel_moves(self, pair, straight_line):
        all_straight_lines = self.game.board.straight_lines
        line_index = all_straight_lines.index(straight_line)

        # Ensure the indices for parallel lines are within bounds
        first_parallel_line = all_straight_lines[line_index - 1] if line_index > 0 else None
        second_parallel_line = all_straight_lines[line_index + 1] if line_index < len(all_straight_lines) - 1 else None

        valid_parallel_moves = []

        # Function to check if a cell is adjacent to any ball in the pair
        def is_adjacent_to_pair(cell):
            return any(cell in self.adjacent_cells_dict[ball] for ball in pair)

        # Check parallel lines if they exist
        for parallel_line in [first_parallel_line, second_parallel_line]:
            if parallel_line:
                for i in range(len(parallel_line) - 1):
                    adjacent_ball1, adjacent_ball2 = parallel_line[i], parallel_line[i + 1]
                    # Check if both cells are empty and adjacent to one of the balls in the pair
                    if (self.game.board.get_cell(adjacent_ball1[0], adjacent_ball1[1]) == 0 and 
                        self.game.board.get_cell(adjacent_ball2[0], adjacent_ball2[1]) == 0 and 
                        is_adjacent_to_pair(adjacent_ball1) and 
                        is_adjacent_to_pair(adjacent_ball2)):
                            valid_parallel_moves.append((adjacent_ball1, adjacent_ball2))

        return valid_parallel_moves

    def get_legitimate_one_ball_moves(self, player_positions, adjacent_cells=None):
        
        legitimate_one_ball_moves = {}


        # Set adjacent cells to our predefined dictionary
        if adjacent_cells is None:
            adjacent_cells = self.adjacent_cells_dict

        # Now iterate over all player's balls and check if any adjacent cell to that ball is empty - if yes, then it's a legal move
        for ball in player_positions:
            # Get the current list of moves for the ball or an empty list if none are found
            current_moves = legitimate_one_ball_moves.get(ball, [])
            for adjacent_cell in adjacent_cells[ball]:
                if self.game.board.get_cell(adjacent_cell[0], adjacent_cell[1]) == 0:
                    # Append to the list of moves for the ball
                    current_moves.append(adjacent_cell)
            # If there are any moves for the ball, update the dictionary
            if current_moves:
                legitimate_one_ball_moves[ball] = current_moves

        return legitimate_one_ball_moves

    def get_legitimate_two_balls_moves(self, player, player_positions, adjacent_cells=None):
        # Using defaultdict for automatic initialization of missing keys
        legitimate_two_ball_moves = defaultdict(list)

        if adjacent_cells is None:
            adjacent_cells = self.adjacent_cells_dict

        # Finding adjacent pairs of player's balls
        adjacent_pairs = [
            (ball1, ball2) 
            for i, ball1 in enumerate(player_positions) 
            for ball2 in player_positions[i+1:] 
            if ball2 in adjacent_cells[ball1]
        ]

        for adjacent_pair in adjacent_pairs:
            straight_line = self.game.board.pairs_to_straight_lines[adjacent_pair]

            # Mapping each cell to its index within its line for faster access
            indices = {cell: index for index, cell in enumerate(straight_line)}

            # Define a function to check the move validity in a given direction
            def check_move_direction(cell, direction):
                index = indices[cell]
                next_index = index + direction
                # Check if the next index is within bounds
                if 0 <= next_index < len(straight_line):
                    next_cell = straight_line[next_index]
                    cell_value = self.game.board.get_cell(next_cell[0], next_cell[1])
                    # Check the cell value and act accordingly
                    if cell_value == 0:
                        legitimate_two_ball_moves[adjacent_pair].append((next_cell, cell))
                    elif cell_value != player and cell_value is not None:
                        # If the next cell is not empty and not the player's, check the next one
                        next_next_index = next_index + direction
                        if 0 <= next_next_index < len(straight_line):
                            next_next_cell = straight_line[next_next_index]
                            if self.game.board.get_cell(next_next_cell[0],next_next_cell[1]) == 0:
                                legitimate_two_ball_moves[adjacent_pair].append((next_cell, cell))

            # Check both directions for each pair
            check_move_direction(adjacent_pair[0], -1)
            check_move_direction(adjacent_pair[1], 1)

            # Now I also need to come up with a way to find parallel moves
            # So we basically need to check the two straight lines that are adjacent to the line the pair is on 
            # And then on both on those straight lines we need to find two adjacent balls where each of those balls either belongs to 
            # A straight line that one of the original balls lie on
            
            parallel_moves_for_pair = self.parallel_moves(adjacent_pair, straight_line)
            legitimate_two_ball_moves[adjacent_pair].extend(parallel_moves_for_pair)
            
        return dict(legitimate_two_ball_moves, player, player_positions, adjacent_cells=None)


    def get_legitimate_three_balls_moves(self, player, player_positions, adjacent_cells=None):
        # Using defaultdict for automatic initialization of missing keys
        legitimate_three_ball_moves = defaultdict(list)

        # Create the adjacent trios to straight lines dictionary
        adjacent_trios_dict = self.game.board.trios_to_straight_lines

        # Find adjacent trios of player's balls
        adjacent_trios = [
            trio for trio in adjacent_trios_dict
            if all(ball in player_positions for ball in trio)
        ]

        # Define a function to check the move validity in a given direction
        def check_move_direction(trio, direction):
            straight_line = adjacent_trios_dict[trio]
            indices = {cell: index for index, cell in enumerate(straight_line)}

            # Find the indices of the trio in the straight line
            start_index = indices[trio[0]]
            end_index = indices[trio[2]]

            # Determine the indices to check based on the direction
            check_indices = [start_index + direction, end_index + direction]
            
            i = 0
            for index in check_indices:
                
                # Ensure the index is within the bounds of the line
                if 0 <= index < len(straight_line):
                    next_cell = straight_line[index] 
                    cell_value = self.game.board.get_cell(next_cell[0], next_cell[1])
                    if cell_value == 0:
                        if i == 0:
                            legitimate_three_ball_moves[trio].append((next_cell, trio[0],trio[1]))
                        else:
                            legitimate_three_ball_moves[trio].append((next_cell, trio[2],trio[1]))
                    elif cell_value != player and cell_value is not None:
                        # Check the next cell if the first one is an opponent's ball
                        next_index = index + direction
                        if 0 <= next_index < len(straight_line):
                            next_next_cell = straight_line[next_index]
                            if self.game.board.get_cell(next_next_cell[0], next_next_cell[1]) == 0:
                                if i == 0:
                                    legitimate_three_ball_moves[trio].append((next_cell, trio[0],trio[1]))
                                else:
                                    legitimate_three_ball_moves[trio].append((next_cell, trio[2],trio[1]))
                            
                        elif self.game.board.get_cell(next_next_cell[0], next_next_cell[1]) != player and self.game.board.get_cell(next_next_cell[0], next_next_cell[1]) != None:
                            next_next_index = next_index + direction
                            if 0 <= next_index < len(straight_line):
                                next_next_next_cell = straight_line[next_next_index]
                                if self.game.board.get_cell(next_next_next_cell[0], next_next_next_cell[1]) == 0:
                                    if i == 0:
                                        legitimate_three_ball_moves[trio].append((next_cell, trio[0],trio[1]))
                                    else:
                                        legitimate_three_ball_moves[trio].append((next_cell, trio[2],trio[1]))

                i+=1 
        # Check both directions for each trio
        for trio in adjacent_trios:
            check_move_direction(trio, -1)
            check_move_direction(trio, 1)

        return legitimate_three_ball_moves
            
    def get_action_space(self):

        action_space = []
        # here I search for all current player's ball positions
        curr_player_ball_positions = self.find_player_balls(self.game.board,self.current_player)

        # first get all legitimate one ball moves
        one_ball_moves = self.get_legitimate_one_ball_moves(curr_player_ball_positions)

        # then get all 2 ball moves
        two_balls_moves = self.get_legitimate_two_balls_moves(self.current_player, curr_player_ball_positions)

        # then get 3 ball moves
        three_balls_moves = self.get_legitimate_three_balls_moves(self.current_player, curr_player_ball_positions)
        
    def get_current_state(self):
        """Return the current state of the board."""
        return self.game.board.grid

    def make_move(self, balls_start, balls_end):
        """Attempt to make a move and return if the move was successful."""
        return self.game.make_move(balls_start, balls_end)

    def is_game_over(self):
        """Check if the game is over."""
        return any(player.score == 6 for player in self.game.players)

    def get_winner(self):
        """Return the player who has won the game or None if there's no winner yet."""
        for player in self.game.players:
            if player.score == 6:
                return player
        return None
    
    def sort_balls(self, balls_start, balls_end):
        if len(balls_start) > 1:
            set1 = set(balls_start)
            set2 = set(balls_end)
            common_elements = set1.intersection(set2)
            if len(common_elements) > 0:
                # Find the unique elements
                unique_start = list(set1 - set2)[0]
                unique_end = list(set2 - set1)[0]

                def distance_to_unique_end(ball):
                    return abs(ball[0] - unique_end[0]) + abs(ball[1] - unique_end[1])

                sorted_balls_start = sorted([ball for ball in balls_start if ball != unique_start], key=distance_to_unique_end)
                sorted_balls_end = sorted([ball for ball in balls_end if ball != unique_end], key=distance_to_unique_end)
            
                sorted_balls_start.append(unique_start)
                sorted_balls_end.insert(0, unique_end)
            
                return sorted_balls_start, sorted_balls_end
            else:
                if len(balls_start) == 2:
                    direction = (balls_start[0][0] - balls_start[1][0], balls_start[0][1] - balls_start[1][1])
                    if (balls_end[0][0] - balls_end[1][0], balls_end[0][1] - balls_end[1][1]) == direction:
                        return balls_start, balls_end 
                    else:
                        balls_end = balls_end[::-1]
                        return balls_start, balls_end
                else:
                    # if there are 3 balls moving parallel we need to sort them with the one having middle row/ column in the middle
                    # and then sort balls_end also with the middle one in the middle and the first one being closest to the first one in balls_start
                    sorted_balls_start = sorted(balls_start, key=lambda x: (x[0], x[1]))
                    middle_ball = sorted_balls_start[1]
                    sorted_balls_end = sorted(balls_end, key=lambda x: abs(x[0] - middle_ball[0]) + abs(x[1] - middle_ball[1]))
                    
                    # Ensure that the first ball in balls_end is closest to the first ball in balls_start
                    if abs(sorted_balls_end[0][0] - sorted_balls_start[0][0]) + abs(sorted_balls_end[0][1] - sorted_balls_start[0][1]) > \
                    abs(sorted_balls_end[-1][0] - sorted_balls_start[0][0]) + abs(sorted_balls_end[-1][1] - sorted_balls_start[0][1]):
                        sorted_balls_end = sorted_balls_end[::-1]

                    return sorted_balls_start, sorted_balls_end


        else:
            return balls_start, balls_end
        
