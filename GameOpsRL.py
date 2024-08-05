from GameRL import GameRL
from Player import Player
from Board import Board 
from collections import defaultdict
import torch
import copy

class GameOpsRL:
    def __init__(self, player1, player2, max_moves = None):
        self.game = GameRL(player1, player2)
        self.current_player = self.game.current_player.color   
        self.adjacent_cells_dict = self.all_adjacent_balls(self.game.board.grid)
        self.max_moves = max_moves


    def reset(self):
        self.game.initialize_game()
        return self.get_current_state()

    def step(self, action):
        balls_start = action['start']
        balls_end = action['end']
        move_type = action['type']

        if isinstance(balls_start[0], tuple):
            balls_start, balls_end = self.sort_balls(balls_start, balls_end)

        if self.make_move(balls_start, balls_end) is False:
            # Return the current state unchanged and False for both state change and done
            return self.get_current_state(), False, False

        # Check if the game is over
        done = self.is_game_over()

        # Return the new state, True for successful move, and done status
        return self.get_current_state(), True, done
    
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
                        adjacent_cells.extend([(index, col_n - 1), (index + 1, col_n), (index + 1, col_n + 1)])
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
        adjacent_duos_dict = self.game.board.pairs_to_straight_lines

        # Find adjacent trios of player's balls
        adjacent_pairs = [
            duo for duo in adjacent_duos_dict
            if all(ball in player_positions for ball in duo)
        ]
        adjacent_pairs = list(dict.fromkeys(tuple(sorted(duo)) for duo in adjacent_duos_dict if all(ball in player_positions for ball in duo)))
        # print('Adjacent pairs:', adjacent_pairs)
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
                        elif next_next_index == 0 or next_next_index == len(straight_line):
                            legitimate_two_ball_moves[adjacent_pair].append((next_cell, cell))
                            
            # Check both directions for each pair
            check_move_direction(adjacent_pair[0], -1)
            check_move_direction(adjacent_pair[1], 1)

            # Now I also need to come up with a way to find parallel moves
            # So we basically need to check the two straight lines that are adjacent to the line the pair is on 
            # And then on both on those straight lines we need to find two adjacent balls where each of those balls either belongs to 
            # A straight line that one of the original balls lie on

            def parallel_moves(pair, straight_line):

                # find the two cells which are adjacent to both the balls in the pair
                adjacent_cells = []
                cell1 = pair[0]
                cell2 = pair[1]
                
                for adj_cell in self.adjacent_cells_dict[cell1]:
                    if adj_cell in self.adjacent_cells_dict[cell2]:
                        adjacent_cells.append(adj_cell)
                    
                # now we need to check whether the adjacent cells are empty, and if yes, then also one cell to the right and left since a pair
                # has 4 theorically possible parallel moves
                parralel_lines = self.return_parallel_lines(straight_line)
                # print(f' the lines we will check are : {parralel_lines} for pair: {pair}')
                for cell in adjacent_cells:


                    if self.game.board.get_cell(cell[0], cell[1]) == 0:
                        # print(f'checking index for cell: {cell}')
                        # if the cell is empty, then we need to check the cells to the right and left of it but first check on which
                        # of the parallel lines it lies

                        if len(parralel_lines) == 2:
                            for line in parralel_lines:
                                # print(f'checking parallel line: {line}')
                                if cell in line:
                                    parallel_line = line
                                    break
                        else:
                            if cell in parralel_lines:
                                    parallel_line = parralel_lines
                                    break             
                        # now we need to find the index of the cell in the parallel line

                        
                        index = parallel_line.index(cell)
                        # now we need to check the cells to the right and left of the cell
                        if index > 0:
                            if self.game.board.get_cell(parallel_line[index-1][0], parallel_line[index-1][1]) == 0:
                                legitimate_two_ball_moves[pair].append((parallel_line[index-1], cell))

                        if index < len(parallel_line) - 1:
                            if self.game.board.get_cell(parallel_line[index+1][0], parallel_line[index+1][1]) == 0:
                                legitimate_two_ball_moves[pair].append((cell, parallel_line[index+1]))


            # let's add parallel moves now 
            parallel_moves(adjacent_pair, straight_line)


        return legitimate_two_ball_moves


    def return_parallel_lines(self, straight_line):

        parallel_lines_one = [[(4,0),(5,0),(6,0),(7,0),(8,0)],
                              [(3,0),(4,1),(5,1),(6,1),(7,1),(8,1)],
                              [(2,0),(3,1),(4,2),(5,2),(6,2),(7,2),(8,2)],
                              [(1,0),(2,1),(3,2),(4,3),(5,3),(6,3),(7,3),(8,3)],
                              [(0,0), (1,1),(2,2),(3,3),(4,4),(5,4),(6,4),(7,4),(8,4)],
                              [(0,1),(1,2),(2,3),(3,4),(4,5),(5,5),(6,5),(7,5)],
                              [(0,2),(1,3),(2,4),(3,5),(4,6),(5,6),(6,6)],
                              [(0,3),(1,4),(2,5),(3,6),(4,7),(5,7)],
                              [(0,4),(1,5),(2,6),(3,7),(4,8)]]
        
        parallel_lines_two = [[(0,0),(0,1),(0,2),(0,3),(0,4)],
                              [(1,0),(1,1),(1,2),(1,3),(1,4),(1,5)],
                              [(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6)],
                              [(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7)],
                              [(4,0), (4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8)],
                              [(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7)],
                              [(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)],
                              [(7,0),(7,1),(7,2),(7,3),(7,4),(7,5)],
                              [(8,0),(8,1),(8,2),(8,3),(8,4)]]
        
        parallel_lines_three = [[(4,8),(5,7),(6,6),(7,5),(8,4)],
                              [(3,7),(4,7),(5,6),(6,5),(7,4),(8,3)],
                              [(2,6),(3,6),(4,6),(5,5),(6,4),(7,3),(8,2)],
                              [(1,5),(2,5),(3,5),(4,5),(5,4),(6,3),(7,2),(8,1)],
                              [(0,4), (1,4),(2,4),(3,4),(4,4),(5,3),(6,2),(7,1),(8,0)],
                              [(0,3),(1,3),(2,3),(3,3),(4,3),(5,2),(6,1),(7,0)],
                              [(0,2),(1,2),(2,2),(3,2),(4,2),(5,1),(6,0)],
                              [(0,1),(1,1),(2,1),(3,1),(4,1),(5,0)],
                              [(0,0),(1,0),(2,0),(3,0),(4,0)]]
        
        if straight_line in parallel_lines_one:
            straight_line_index = parallel_lines_one.index(straight_line)

            if straight_line_index > 0 and straight_line_index < 8:
                return parallel_lines_one[straight_line_index-1], parallel_lines_one[straight_line_index+1]

            if straight_line_index == 0:
                    return parallel_lines_one[straight_line_index+1]
                
            if straight_line_index == 8:
                    return parallel_lines_one[straight_line_index-1]
        
        if straight_line in parallel_lines_two:
            straight_line_index = parallel_lines_two.index(straight_line)
            if straight_line_index > 0 and straight_line_index < 8:
                return parallel_lines_two[straight_line_index-1], parallel_lines_two[straight_line_index+1]

            if straight_line_index == 0:
                    return parallel_lines_two[straight_line_index+1]
                
            if straight_line_index == 8:
                    return parallel_lines_two[straight_line_index-1]
        
        if straight_line in parallel_lines_three:
            straight_line_index = parallel_lines_three.index(straight_line)
            if straight_line_index > 0 and straight_line_index < 8:
                return parallel_lines_three[straight_line_index-1], parallel_lines_three[straight_line_index+1]

            if straight_line_index == 0:
                    return parallel_lines_three[straight_line_index+1]
                
            if straight_line_index == 8:
                    return parallel_lines_three[straight_line_index-1]
        

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
                            
                            elif self.game.board.get_cell(next_next_cell[0], next_next_cell[1]) != player and self.game.board.get_cell(next_next_cell[0], next_next_cell[1]) is not None:
                                next_next_index = next_index + direction
                                if 0 <= next_next_index < len(straight_line):
                                    next_next_next_cell = straight_line[next_next_index]
                                    if self.game.board.get_cell(next_next_next_cell[0], next_next_next_cell[1]) == 0:
                                        if i == 0:
                                            legitimate_three_ball_moves[trio].append((next_cell, trio[0],trio[1]))
                                        else:
                                            legitimate_three_ball_moves[trio].append((next_cell, trio[2],trio[1]))

                                elif next_next_index == 0 or next_next_index == len(straight_line):
                                    if i == 0:
                                        legitimate_three_ball_moves[trio].append((next_cell, trio[0],trio[1]))
                                    else:
                                        legitimate_three_ball_moves[trio].append((next_cell, trio[2],trio[1]))
                        elif next_index == 0 or next_index == len(straight_line):
                            if i == 0:
                                legitimate_three_ball_moves[trio].append((next_cell, trio[0],trio[1]))
                            else:
                                legitimate_three_ball_moves[trio].append((next_cell, trio[2],trio[1]))
                i+=1 
        # Check both directions for each trio
        for trio in adjacent_trios:
            check_move_direction(trio, -1)
            check_move_direction(trio, 1)

        def sort_trio_and_duo(trio, duo):
            set1 = set(trio)
            set2 = set(duo)
            common_elements = set1.intersection(set2)
            
            if len(common_elements) > 0:
                # Find the unique elements
                unique_trio = list(set1 - set2)[0]
                unique_duo = list(set2 - set1)[0]

                def distance_to_unique_duo(ball):
                    return abs(ball[0] - unique_duo[0]) + abs(ball[1] - unique_duo[1])

                sorted_trio = sorted([ball for ball in trio if ball != unique_trio], key=distance_to_unique_duo)
                sorted_trio.append(unique_trio)
                
                sorted_duo = list(duo)
                sorted_duo.remove(unique_duo)
                sorted_duo.insert(0, unique_duo)
            else:
                # For parallel moves
                sorted_trio = sorted(trio, key=lambda x: (x[0], x[1]))
                
                # Determine the direction of movement
                direction = (duo[0][0] - trio[0][0], duo[0][1] - trio[0][1])
                
                # Sort duo in the same order as trio
                sorted_duo = sorted(duo, key=lambda x: (x[0] - direction[0], x[1] - direction[1]))
            
            return sorted_trio, sorted_duo

        def parallel_moves(trio):
                
                # the difference for parallel moves between two and three balls is that either we find 2 cells on adjacent straight line that are adjacent to the trio
                # or we can find 2 cell on adjacent straight line that is adjacent to the middle ball of the trio 



                straight_line = adjacent_trios_dict[trio]
                # find the two cells which are adjacent to both the balls in the pair
                adjacent_cells = []
                cell1, cell2, cell3 = trio

                print(f'Trio: {trio}')
                
                for adj_cell in self.adjacent_cells_dict[cell2]:
                    if adj_cell in self.adjacent_cells_dict[cell1] or adj_cell in self.adjacent_cells_dict[cell3]:
                        adjacent_cells.append(adj_cell)
                
                
                adjacent_pairs = []

                # finding pairs among the adjacent cells
                # let's simply check whether they lie on the same straight line

                adjacent_duos_dict = self.game.board.pairs_to_straight_lines

                for i in range(len(adjacent_cells)):
                    for j in range(i+1, len(adjacent_cells)):
                        if adjacent_duos_dict.get((adjacent_cells[i], adjacent_cells[j])) is not None:
                            adjacent_pairs.append((adjacent_cells[i], adjacent_cells[j]))
                        elif adjacent_duos_dict.get((adjacent_cells[j], adjacent_cells[i])) is not None:
                            adjacent_pairs.append((adjacent_cells[j], adjacent_cells[i]))

                # print out the results
                #for duo in adjacent_pairs:
                    #print(f'Duo in adjacent pairs to trio: {duo}')
                                           
                # now we need to check whether the adjacent cells are empty, and if yes, then also one cell to the right and left since a pair
                # has 4 theorically possible parallel moves
                parralel_lines = self.return_parallel_lines(straight_line)
                adjacent_pairs = [sort_trio_and_duo(trio, duo) for duo in adjacent_pairs]

                def check_parallel_line(parralel_lines, duo):

                    cell1 = duo[0]
                    cell2 = duo[1]
                    if self.game.board.get_cell(cell1[0], cell1[1]) == 0 and self.game.board.get_cell(cell2[0], cell2[1]) == 0:
                        
                            # if the cells are empty, then we need to check the cells to the right and left of it but first check on which
                            # of the parallel lines it lies

                            if len(parralel_lines) == 2:
                                for line in parralel_lines:
                                    # print(f'checking parallel line in trio: {line}')
                                    if cell1 in line:
                                        parallel_line = line
                                        break
                            else:
                                if cell1 in parralel_lines:
                                    parallel_line = parralel_lines
                                    # print(f'checking parallel line in trio: {parallel_line}')
                                    
                            
                            #print(f'the line we will check is : {parallel_line} for duo: {duo}')
                            index1 = parallel_line.index(cell1)
                            index2 = parallel_line.index(cell2)
                            # now we need to check the cells to the right and left of the cell
                            if index1 > 0:
                                if self.game.board.get_cell(parallel_line[index1-1][0], parallel_line[index1-1][1]) == 0:
                                    #print(f'true that the cell {parallel_line.index(cell1)} is empty')
                                    #print(f'Appending with {parallel_line[index1-1]}')
                                    legitimate_three_ball_moves[trio].append((parallel_line[index1-1], cell1, cell2))

                            if index2 < len(parallel_line) - 1:
                                if self.game.board.get_cell(parallel_line[index2+1][0], parallel_line[index2+1][1]) == 0:
                                    legitimate_three_ball_moves[trio].append((cell1, cell2, parallel_line[index2+1]))

                for _, sorted_duo in adjacent_pairs:
                    check_parallel_line(parralel_lines, sorted_duo)
                    
        for trio in adjacent_trios:
            parallel_moves(trio)

        return legitimate_three_ball_moves
            
    def get_action_space(self):
        action_space = []
        action_index = 0
        action_details = {}
        action_mask = []

        # Search for all current player's ball positions

        print(f' current player check in get action space: {self.game.current_player.color}')
        curr_player_ball_positions = self.find_player_balls(self.game.board.grid, self.game.current_player.color)

        # print(f'Current player ball positions: {curr_player_ball_positions}')

        # Get all legitimate moves for one, two, and three balls
        one_ball_moves = self.get_legitimate_one_ball_moves(curr_player_ball_positions)

        print(f' current player : {self.game.current_player.color}, curr player ball positions: {curr_player_ball_positions}')
        two_balls_moves = self.get_legitimate_two_balls_moves(self.game.current_player.color, curr_player_ball_positions)
        three_balls_moves = self.get_legitimate_three_balls_moves(self.game.current_player.color, curr_player_ball_positions)

        # Combining all moves into a single action space with unique indices
        for move_type in [one_ball_moves, two_balls_moves, three_balls_moves]:
            for start_position, moves in move_type.items():
                for end_position in moves:
                    # Store each action as a tuple of start and end positions
                    action = (start_position, end_position)
                    action_details[action_index] = {
                        'start': start_position,
                        'end': end_position,
                        'type': len(start_position)  # Depending on the number of balls involved
                        
                    }
                    action_space.append(action_index)
                    action_mask.append(True)  # Mark this action as valid
                    action_index += 1

        # Ensure the action mask has the same length as the action space
        # This might be redundant if actions are guaranteed to be legal once added
        if len(action_mask) < len(action_space):
            action_mask.extend([False] * (len(action_space) - len(action_mask)))

        return action_space, action_details, action_mask
    
    
    def get_current_state(self):
        """Return a deep copy of the current state of the board."""
        current_state = copy.deepcopy(self.game.board.grid)
        current_player = [self.game.current_player.color]  # This is already a new list, so no need for deepcopy
        return current_state, current_player

    def make_move(self, balls_start, balls_end):
        """Attempt to make a move and return if the move was successful."""

        print(f'Making move from {balls_start} to {balls_end}')
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
                    # For 3-ball parallel moves
                    sorted_balls_start = sorted(balls_start, key=lambda x: (x[0], x[1]))
                    
                    # Determine the direction of movement
                    direction = (balls_end[0][0] - balls_start[0][0], balls_end[0][1] - balls_start[0][1])
                    
                    # Sort balls_end in the same order as balls_start
                    sorted_balls_end = sorted(balls_end, key=lambda x: (x[0] - direction[0], x[1] - direction[1]))
                    
                    return sorted_balls_start, sorted_balls_end
        else:
            return balls_start, balls_end



    def encode_action(self, action):
        start = action['start']
        end = action['end']
        move_type = action['type']

        if isinstance(start[0], int):
            start = [start]
            end = [end]
        
        # Example encoding scheme
        start_flat = [item for sublist in start for item in sublist]  # Flatten start
        end_flat = [item for sublist in end for item in sublist]  # Flatten end
        
        # Combine all parts into a single list and then into a tensor
        encoded_action = start_flat + end_flat + [move_type]
        return torch.tensor(encoded_action, dtype=torch.long)
    
    def decode_action(self, encoded_action):
        
        encoded_list = encoded_action.tolist()
        start = [(encoded_list[0], encoded_list[1]), (encoded_list[2], encoded_list[3])]
        end = [(encoded_list[4], encoded_list[5]), (encoded_list[6], encoded_list[7])]
        move_type = encoded_list[8]
        
        return {'start': start, 'end': end, 'type': move_type}