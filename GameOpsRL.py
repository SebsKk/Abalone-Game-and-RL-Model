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
        self.all_actions = self.define_all_actions()


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
            print(f'Invalid move: {balls_start} to {balls_end}')
            # Return the current state unchanged and False for both state change and done
            return self.get_current_state(), False, False

        # Check if the game is over
        done = self.is_game_over()

        # Return the new state, True for successful move, and done status
        return self.get_current_state(), True, done
    
    def simulate_step(self, state, action):
        # Create a deep copy of the current game state
        temp_game = copy.deepcopy(self.game)
        temp_game_ops = copy.deepcopy(self)
        temp_game_ops.game = temp_game

        balls_start = action['start']
        balls_end = action['end']
        move_type = action['type']

        if isinstance(balls_start[0], tuple):
            balls_start, balls_end = temp_game_ops.sort_balls(balls_start, balls_end)

        # Simulate the move on the temporary game state
        if temp_game_ops.make_move(balls_start, balls_end) is False:
            # Return the current state unchanged and False for both state change and done
            return state, False, False

        # Check if the game is over in the simulated state
        done = temp_game_ops.is_game_over()

        # Get the new state from the temporary game
        new_state = temp_game_ops.get_current_state()

        # Return the new state, True for successful move, and done status
        return new_state, True, done
    
    def find_player_balls(self, board, player):
 
        player_positions = []

        #print(f'the board is {board}, player is {player}')
        for row_index, row in enumerate(board):
            for col_index, cell in enumerate(row):
                if cell == player:
                    #print(f'found a player at {row_index, col_index}')
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
                        adjacent_cells.extend([(index, col_n - 1), (index + 1, col_n), (index - 1, col_n - 1), (index + 1, col_n + 1)])
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

                # print(f'Trio: {trio}')
                
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
        action_details = {}
        action_mask = []

        curr_player_ball_positions = self.find_player_balls(self.game.board.grid, self.game.current_player.color)

        one_ball_moves = self.get_legitimate_one_ball_moves(curr_player_ball_positions)
        two_balls_moves = self.get_legitimate_two_balls_moves(self.game.current_player.color, curr_player_ball_positions)
        three_balls_moves = self.get_legitimate_three_balls_moves(self.game.current_player.color, curr_player_ball_positions)

        for move_type in [one_ball_moves, two_balls_moves, three_balls_moves]:
            for start_position, moves in move_type.items():
                for end_position in moves:
                    action = {
                        'start': start_position,
                        'end': end_position,
                        'type': len(start_position) if isinstance(start_position, tuple) and isinstance(start_position[0], tuple) else 1
                    }

                    action_key = self.get_action_key(action)
                    
                    proper_index = self.all_actions[action_key]
                    action_details[proper_index] = action
                    action_space.append(proper_index)
                    action_mask.append(True)
                    
                    
                

        if len(action_mask) < len(self.all_actions):
            action_mask.extend([False] * (len(self.all_actions) - len(action_mask)))

        return action_space, action_details, action_mask

    def get_action_key(self, action):
        """
        Helper method to create a key for the action that can be used to look it up in self.all_actions.
        The key should match the format used in self.all_actions
        """

        return (action['start'], action['end'], action['type'])
    

    def get_current_state(self):
        """Return a deep copy of the current state of the board."""
        current_state = copy.deepcopy(self.game.board.grid)
        current_player = [self.game.current_player.color]  
        return current_state, current_player

    def make_move(self, balls_start, balls_end):
        """Attempt to make a move and return if the move was successful."""

        #print(f'Making move from {balls_start} to {balls_end}')
        return self.game.make_move(balls_start, balls_end)

    def is_game_over(self):
        """Check if the game is over."""
        for player in self.game.players:
            if player.score == 6:
                print(f"Game over! Player {player.color} has won by reaching 6 points.")
                return True
        
    
        return False

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
    

    def define_all_actions(self):

        all_actions = {((0, 0), (0, 1), 1): 0, ((0, 0), (1, 0), 1): 1, ((0, 0), (1, 1), 1): 2, ((0, 1), (0, 2), 1): 3, ((0, 1), (0, 0), 1): 4, ((0, 1), (1, 1), 1): 5, ((0, 1), (1, 2), 1): 6, ((0, 2), (0, 3), 1): 7, ((0, 2), (0, 1), 1): 8, ((0, 2), (1, 2), 1): 9, ((0, 2), (1, 3), 1): 10, ((0, 3), (0, 4), 1): 11, ((0, 3), (0, 2), 1): 12, ((0, 3), (1, 3), 1): 13, ((0, 3), (1, 4), 1): 14, ((0, 4), (0, 3), 1): 15, ((0, 4), (1, 4), 1): 16, ((0, 4), (1, 5), 1): 17, ((1, 0), (1, 1), 1): 18, ((1, 0), (2, 0), 1): 19, ((1, 0), (0, 0), 1): 20, ((1, 0), (2, 1), 1): 21, ((1, 1), (1, 2), 1): 22, ((1, 1), (1, 0), 1): 23, ((1, 1), (2, 1), 1): 24, ((1, 1), (2, 2), 1): 25, ((1, 1), (0, 1), 1): 26, ((1, 1), (0, 0), 1): 27, ((1, 2), (1, 3), 1): 28, ((1, 2), (1, 1), 1): 29, ((1, 2), (2, 2), 1): 30, ((1, 2), (2, 3), 1): 31, ((1, 2), (0, 2), 1): 32, ((1, 2), (0, 1), 1): 33, ((1, 3), (1, 4), 1): 34, ((1, 3), (1, 2), 1): 35, ((1, 3), (2, 3), 1): 36, ((1, 3), (2, 4), 1): 37, ((1, 3), (0, 3), 1): 38, ((1, 3), (0, 2), 1): 39, ((1, 4), (1, 5), 1): 40, ((1, 4), (1, 3), 1): 41, ((1, 4), (2, 4), 1): 42, ((1, 4), (2, 5), 1): 43, ((1, 4), (0, 4), 1): 44, ((1, 4), (0, 3), 1): 45, ((1, 5), (1, 4), 1): 46, ((1, 5), (2, 5), 1): 47, ((1, 5), (0, 4), 1): 48, ((1, 5), (2, 6), 1): 49, ((2, 0), (2, 1), 1): 50, ((2, 0), (3, 0), 1): 51, ((2, 0), (1, 0), 1): 52, ((2, 0), (3, 1), 1): 53, ((2, 1), (2, 2), 1): 54, ((2, 1), (2, 0), 1): 55, ((2, 1), (3, 1), 1): 56, ((2, 1), (3, 2), 1): 57, ((2, 1), (1, 1), 1): 58, ((2, 1), (1, 0), 1): 59, ((2, 2), (2, 3), 1): 60, ((2, 2), (2, 1), 1): 61, ((2, 2), (3, 2), 1): 62, ((2, 2), (3, 3), 1): 63, ((2, 2), (1, 2), 1): 64, ((2, 2), (1, 1), 1): 65, ((2, 3), (2, 4), 1): 66, ((2, 3), (2, 2), 1): 67, ((2, 3), (3, 3), 1): 68, ((2, 3), (3, 4), 1): 69, ((2, 3), (1, 3), 1): 70, ((2, 3), (1, 2), 1): 71, ((2, 4), (2, 5), 1): 72, ((2, 4), (2, 3), 1): 73, ((2, 4), (3, 4), 1): 74, ((2, 4), (3, 5), 1): 75, ((2, 4), (1, 4), 1): 76, ((2, 4), (1, 3), 1): 77, ((2, 5), (2, 6), 1): 78, ((2, 5), (2, 4), 1): 79, ((2, 5), (3, 5), 1): 80, ((2, 5), (3, 6), 1): 81, ((2, 5), (1, 5), 1): 82, ((2, 5), (1, 4), 1): 83, ((2, 6), (2, 5), 1): 84, ((2, 6), (3, 6), 1): 85, ((2, 6), (1, 5), 1): 86, ((2, 6), (3, 7), 1): 87, ((3, 0), (3, 1), 1): 88, ((3, 0), (4, 0), 1): 89, ((3, 0), (2, 0), 1): 90, ((3, 0), (4, 1), 1): 91, ((3, 1), (3, 2), 1): 92, ((3, 1), (3, 0), 1): 93, ((3, 1), (4, 1), 1): 94, ((3, 1), (4, 2), 1): 95, ((3, 1), (2, 1), 1): 96, ((3, 1), (2, 0), 1): 97, ((3, 2), (3, 3), 1): 98, ((3, 2), (3, 1), 1): 99, ((3, 2), (4, 2), 1): 100, ((3, 2), (4, 3), 1): 101, ((3, 2), (2, 2), 1): 102, ((3, 2), (2, 1), 1): 103, ((3, 3), (3, 4), 1): 104, ((3, 3), (3, 2), 1): 105, ((3, 3), (4, 3), 1): 106, ((3, 3), (4, 4), 1): 107, ((3, 3), (2, 3), 1): 108, ((3, 3), (2, 2), 1): 109, ((3, 4), (3, 5), 1): 110, ((3, 4), (3, 3), 1): 111, ((3, 4), (4, 4), 1): 112, ((3, 4), (4, 5), 1): 113, ((3, 4), (2, 4), 1): 114, ((3, 4), (2, 3), 1): 115, ((3, 5), (3, 6), 1): 116, ((3, 5), (3, 4), 1): 117, ((3, 5), (4, 5), 1): 118, ((3, 5), (4, 6), 1): 119, ((3, 5), (2, 5), 1): 120, ((3, 5), (2, 4), 1): 121, ((3, 6), (3, 7), 1): 122, ((3, 6), (3, 5), 1): 123, ((3, 6), (4, 6), 1): 124, ((3, 6), (4, 7), 1): 125, ((3, 6), (2, 6), 1): 126, ((3, 6), (2, 5), 1): 127, ((3, 7), (3, 6), 1): 128, ((3, 7), (4, 7), 1): 129, ((3, 7), (2, 6), 1): 130, ((3, 7), (4, 8), 1): 131, ((4, 0), (4, 1), 1): 132, ((4, 0), (5, 0), 1): 133, ((4, 0), (3, 0), 1): 134, ((4, 1), (4, 2), 1): 135, ((4, 1), (4, 0), 1): 136, ((4, 1), (5, 1), 1): 137, ((4, 1), (5, 0), 1): 138, ((4, 1), (3, 1), 1): 139, ((4, 1), (3, 0), 1): 140, ((4, 2), (4, 3), 1): 141, ((4, 2), (4, 1), 1): 142, ((4, 2), (5, 2), 1): 143, ((4, 2), (5, 1), 1): 144, ((4, 2), (3, 2), 1): 145, ((4, 2), (3, 1), 1): 146, ((4, 3), (4, 4), 1): 147, ((4, 3), (4, 2), 1): 148, ((4, 3), (5, 3), 1): 149, ((4, 3), (5, 2), 1): 150, ((4, 3), (3, 3), 1): 151, ((4, 3), (3, 2), 1): 152, ((4, 4), (4, 5), 1): 153, ((4, 4), (4, 3), 1): 154, ((4, 4), (5, 4), 1): 155, ((4, 4), (5, 3), 1): 156, ((4, 4), (3, 4), 1): 157, ((4, 4), (3, 3), 1): 158, ((4, 5), (4, 6), 1): 159, ((4, 5), (4, 4), 1): 160, ((4, 5), (5, 5), 1): 161, ((4, 5), (5, 4), 1): 162, ((4, 5), (3, 5), 1): 163, ((4, 5), (3, 4), 1): 164, ((4, 6), (4, 7), 1): 165, ((4, 6), (4, 5), 1): 166, ((4, 6), (5, 6), 1): 167, ((4, 6), (5, 5), 1): 168, ((4, 6), (3, 6), 1): 169, ((4, 6), (3, 5), 1): 170, ((4, 7), (4, 8), 1): 171, ((4, 7), (4, 6), 1): 172, ((4, 7), (5, 7), 1): 173, ((4, 7), (5, 6), 1): 174, ((4, 7), (3, 7), 1): 175, ((4, 7), (3, 6), 1): 176, ((4, 8), (4, 7), 1): 177, ((4, 8), (5, 7), 1): 178, ((4, 8), (3, 7), 1): 179, ((5, 0), (5, 1), 1): 180, ((5, 0), (6, 0), 1): 181, ((5, 0), (4, 0), 1): 182, ((5, 0), (4, 1), 1): 183, ((5, 1), (5, 2), 1): 184, ((5, 1), (5, 0), 1): 185, ((5, 1), (6, 1), 1): 186, ((5, 1), (6, 0), 1): 187, ((5, 1), (4, 1), 1): 188, ((5, 1), (4, 2), 1): 189, ((5, 2), (5, 3), 1): 190, ((5, 2), (5, 1), 1): 191, ((5, 2), (6, 2), 1): 192, ((5, 2), (6, 1), 1): 193, ((5, 2), (4, 2), 1): 194, ((5, 2), (4, 3), 1): 195, ((5, 3), (5, 4), 1): 196, ((5, 3), (5, 2), 1): 197, ((5, 3), (6, 3), 1): 198, ((5, 3), (6, 2), 1): 199, ((5, 3), (4, 3), 1): 200, ((5, 3), (4, 4), 1): 201, ((5, 4), (5, 5), 1): 202, ((5, 4), (5, 3), 1): 203, ((5, 4), (6, 4), 1): 204, ((5, 4), (6, 3), 1): 205, ((5, 4), (4, 4), 1): 206, ((5, 4), (4, 5), 1): 207, ((5, 5), (5, 6), 1): 208, ((5, 5), (5, 4), 1): 209, ((5, 5), (6, 5), 1): 210, ((5, 5), (6, 4), 1): 211, ((5, 5), (4, 5), 1): 212, ((5, 5), (4, 6), 1): 213, ((5, 6), (5, 7), 1): 214, ((5, 6), (5, 5), 1): 215, ((5, 6), (6, 6), 1): 216, ((5, 6), (6, 5), 1): 217, ((5, 6), (4, 6), 1): 218, ((5, 6), (4, 7), 1): 219, ((5, 7), (5, 6), 1): 220, ((5, 7), (6, 6), 1): 221, ((5, 7), (4, 7), 1): 222, ((5, 7), (4, 8), 1): 223, ((6, 0), (6, 1), 1): 224, ((6, 0), (7, 0), 1): 225, ((6, 0), (5, 0), 1): 226, ((6, 0), (5, 1), 1): 227, ((6, 1), (6, 2), 1): 228, ((6, 1), (6, 0), 1): 229, ((6, 1), (7, 1), 1): 230, ((6, 1), (7, 0), 1): 231, ((6, 1), (5, 1), 1): 232, ((6, 1), (5, 2), 1): 233, ((6, 2), (6, 3), 1): 234, ((6, 2), (6, 1), 1): 235, ((6, 2), (7, 2), 1): 236, ((6, 2), (7, 1), 1): 237, ((6, 2), (5, 2), 1): 238, ((6, 2), (5, 3), 1): 239, ((6, 3), (6, 4), 1): 240, ((6, 3), (6, 2), 1): 241, ((6, 3), (7, 3), 1): 242, ((6, 3), (7, 2), 1): 243, ((6, 3), (5, 3), 1): 244, ((6, 3), (5, 4), 1): 245, ((6, 4), (6, 5), 1): 246, ((6, 4), (6, 3), 1): 247, ((6, 4), (7, 4), 1): 248, ((6, 4), (7, 3), 1): 249, ((6, 4), (5, 4), 1): 250, ((6, 4), (5, 5), 1): 251, ((6, 5), (6, 6), 1): 252, ((6, 5), (6, 4), 1): 253, ((6, 5), (7, 5), 1): 254, ((6, 5), (7, 4), 1): 255, ((6, 5), (5, 5), 1): 256, ((6, 5), (5, 6), 1): 257, ((6, 6), (6, 5), 1): 258, ((6, 6), (7, 5), 1): 259, ((6, 6), (5, 6), 1): 260, ((6, 6), (5, 7), 1): 261, ((7, 0), (7, 1), 1): 262, ((7, 0), (8, 0), 1): 263, ((7, 0), (6, 0), 1): 264, ((7, 0), (6, 1), 1): 265, ((7, 1), (7, 2), 1): 266, ((7, 1), (7, 0), 1): 267, ((7, 1), (8, 1), 1): 268, ((7, 1), (8, 0), 1): 269, ((7, 1), (6, 1), 1): 270, ((7, 1), (6, 2), 1): 271, ((7, 2), (7, 3), 1): 272, ((7, 2), (7, 1), 1): 273, ((7, 2), (8, 2), 1): 274, ((7, 2), (8, 1), 1): 275, ((7, 2), (6, 2), 1): 276, ((7, 2), (6, 3), 1): 277, ((7, 3), (7, 4), 1): 278, ((7, 3), (7, 2), 1): 279, ((7, 3), (8, 3), 1): 280, ((7, 3), (8, 2), 1): 281, ((7, 3), (6, 3), 1): 282, ((7, 3), (6, 4), 1): 283, ((7, 4), (7, 5), 1): 284, ((7, 4), (7, 3), 1): 285, ((7, 4), (8, 4), 1): 286, ((7, 4), (8, 3), 1): 287, ((7, 4), (6, 4), 1): 288, ((7, 4), (6, 5), 1): 289, ((7, 5), (7, 4), 1): 290, ((7, 5), (8, 4), 1): 291, ((7, 5), (6, 5), 1): 292, ((7, 5), (6, 6), 1): 293, ((8, 0), (8, 1), 1): 294, ((8, 0), (7, 1), 1): 295, ((8, 0), (7, 0), 1): 296, ((8, 1), (8, 2), 1): 297, ((8, 1), (8, 0), 1): 298, ((8, 1), (7, 1), 1): 299, ((8, 1), (7, 2), 1): 300, ((8, 2), (8, 3), 1): 301, ((8, 2), (8, 1), 1): 302, ((8, 2), (7, 2), 1): 303, ((8, 2), (7, 3), 1): 304, ((8, 3), (8, 4), 1): 305, ((8, 3), (8, 2), 1): 306, ((8, 3), (7, 3), 1): 307, ((8, 3), (7, 4), 1): 308, ((8, 4), (8, 3), 1): 309, ((8, 4), (7, 4), 1): 310, ((8, 4), (7, 5), 1): 311, (((4, 0), (5, 0)), ((6, 0), (5, 0)), 2): 312, (((5, 0), (6, 0)), ((4, 0), (5, 0)), 2): 313, (((5, 0), (6, 0)), ((7, 0), (6, 0)), 2): 314, (((6, 0), (7, 0)), ((5, 0), (6, 0)), 2): 315, (((6, 0), (7, 0)), ((8, 0), (7, 0)), 2): 316, (((7, 0), (8, 0)), ((6, 0), (7, 0)), 2): 317, (((3, 0), (4, 1)), ((5, 1), (4, 1)), 2): 318, (((3, 0), (4, 1)), ((2, 0), (3, 1)), 2): 319, (((3, 0), (4, 1)), ((3, 1), (4, 2)), 2): 320, (((3, 0), (4, 1)), ((4, 0), (5, 0)), 2): 321, (((4, 1), (5, 1)), ((3, 0), (4, 1)), 2): 322, (((4, 1), (5, 1)), ((6, 1), (5, 1)), 2): 323, (((4, 1), (5, 1)), ((3, 1), (4, 2)), 2): 324, (((4, 1), (5, 1)), ((4, 2), (5, 2)), 2): 325, (((4, 1), (5, 1)), ((4, 0), (5, 0)), 2): 326, (((4, 1), (5, 1)), ((5, 0), (6, 0)), 2): 327, (((5, 1), (6, 1)), ((4, 1), (5, 1)), 2): 328, (((5, 1), (6, 1)), ((7, 1), (6, 1)), 2): 329, (((5, 1), (6, 1)), ((4, 2), (5, 2)), 2): 330, (((5, 1), (6, 1)), ((5, 2), (6, 2)), 2): 331, (((5, 1), (6, 1)), ((5, 0), (6, 0)), 2): 332, (((5, 1), (6, 1)), ((6, 0), (7, 0)), 2): 333, (((6, 1), (7, 1)), ((5, 1), (6, 1)), 2): 334, (((6, 1), (7, 1)), ((8, 1), (7, 1)), 2): 335, (((6, 1), (7, 1)), ((5, 2), (6, 2)), 2): 336, (((6, 1), (7, 1)), ((6, 2), (7, 2)), 2): 337, (((6, 1), (7, 1)), ((6, 0), (7, 0)), 2): 338, (((6, 1), (7, 1)), ((7, 0), (8, 0)), 2): 339, (((7, 1), (8, 1)), ((6, 1), (7, 1)), 2): 340, (((7, 1), (8, 1)), ((6, 2), (7, 2)), 2): 341, (((7, 1), (8, 1)), ((7, 2), (8, 2)), 2): 342, (((7, 1), (8, 1)), ((7, 0), (8, 0)), 2): 343, (((2, 0), (3, 1)), ((4, 2), (3, 1)), 2): 344, (((2, 0), (3, 1)), ((1, 0), (2, 1)), 2): 345, (((2, 0), (3, 1)), ((2, 1), (3, 2)), 2): 346, (((2, 0), (3, 1)), ((3, 0), (4, 1)), 2): 347, (((3, 1), (4, 2)), ((2, 0), (3, 1)), 2): 348, (((3, 1), (4, 2)), ((5, 2), (4, 2)), 2): 349, (((3, 1), (4, 2)), ((2, 1), (3, 2)), 2): 350, (((3, 1), (4, 2)), ((3, 2), (4, 3)), 2): 351, (((3, 1), (4, 2)), ((3, 0), (4, 1)), 2): 352, (((3, 1), (4, 2)), ((4, 1), (5, 1)), 2): 353, (((4, 2), (5, 2)), ((3, 1), (4, 2)), 2): 354, (((4, 2), (5, 2)), ((6, 2), (5, 2)), 2): 355, (((4, 2), (5, 2)), ((3, 2), (4, 3)), 2): 356, (((4, 2), (5, 2)), ((4, 3), (5, 3)), 2): 357, (((4, 2), (5, 2)), ((4, 1), (5, 1)), 2): 358, (((4, 2), (5, 2)), ((5, 1), (6, 1)), 2): 359, (((5, 2), (6, 2)), ((4, 2), (5, 2)), 2): 360, (((5, 2), (6, 2)), ((7, 2), (6, 2)), 2): 361, (((5, 2), (6, 2)), ((4, 3), (5, 3)), 2): 362, (((5, 2), (6, 2)), ((5, 3), (6, 3)), 2): 363, (((5, 2), (6, 2)), ((5, 1), (6, 1)), 2): 364, (((5, 2), (6, 2)), ((6, 1), (7, 1)), 2): 365, (((6, 2), (7, 2)), ((5, 2), (6, 2)), 2): 366, (((6, 2), (7, 2)), ((8, 2), (7, 2)), 2): 367, (((6, 2), (7, 2)), ((5, 3), (6, 3)), 2): 368, (((6, 2), (7, 2)), ((6, 3), (7, 3)), 2): 369, (((6, 2), (7, 2)), ((6, 1), (7, 1)), 2): 370, (((6, 2), (7, 2)), ((7, 1), (8, 1)), 2): 371, (((7, 2), (8, 2)), ((6, 2), (7, 2)), 2): 372, (((7, 2), (8, 2)), ((6, 3), (7, 3)), 2): 373, (((7, 2), (8, 2)), ((7, 3), (8, 3)), 2): 374, (((7, 2), (8, 2)), ((7, 1), (8, 1)), 2): 375, (((1, 0), (2, 1)), ((3, 2), (2, 1)), 2): 376, (((1, 0), (2, 1)), ((0, 0), (1, 1)), 2): 377, (((1, 0), (2, 1)), ((1, 1), (2, 2)), 2): 378, (((1, 0), (2, 1)), ((2, 0), (3, 1)), 2): 379, (((2, 1), (3, 2)), ((1, 0), (2, 1)), 2): 380, (((2, 1), (3, 2)), ((4, 3), (3, 2)), 2): 381, (((2, 1), (3, 2)), ((1, 1), (2, 2)), 2): 382, (((2, 1), (3, 2)), ((2, 2), (3, 3)), 2): 383, (((2, 1), (3, 2)), ((2, 0), (3, 1)), 2): 384, (((2, 1), (3, 2)), ((3, 1), (4, 2)), 2): 385, (((3, 2), (4, 3)), ((2, 1), (3, 2)), 2): 386, (((3, 2), (4, 3)), ((5, 3), (4, 3)), 2): 387, (((3, 2), (4, 3)), ((2, 2), (3, 3)), 2): 388, (((3, 2), (4, 3)), ((3, 3), (4, 4)), 2): 389, (((3, 2), (4, 3)), ((3, 1), (4, 2)), 2): 390, (((3, 2), (4, 3)), ((4, 2), (5, 2)), 2): 391, (((4, 3), (5, 3)), ((3, 2), (4, 3)), 2): 392, (((4, 3), (5, 3)), ((6, 3), (5, 3)), 2): 393, (((4, 3), (5, 3)), ((3, 3), (4, 4)), 2): 394, (((4, 3), (5, 3)), ((4, 4), (5, 4)), 2): 395, (((4, 3), (5, 3)), ((4, 2), (5, 2)), 2): 396, (((4, 3), (5, 3)), ((5, 2), (6, 2)), 2): 397, (((5, 3), (6, 3)), ((4, 3), (5, 3)), 2): 398, (((5, 3), (6, 3)), ((7, 3), (6, 3)), 2): 399, (((5, 3), (6, 3)), ((4, 4), (5, 4)), 2): 400, (((5, 3), (6, 3)), ((5, 4), (6, 4)), 2): 401, (((5, 3), (6, 3)), ((5, 2), (6, 2)), 2): 402, (((5, 3), (6, 3)), ((6, 2), (7, 2)), 2): 403, (((6, 3), (7, 3)), ((5, 3), (6, 3)), 2): 404, (((6, 3), (7, 3)), ((8, 3), (7, 3)), 2): 405, (((6, 3), (7, 3)), ((5, 4), (6, 4)), 2): 406, (((6, 3), (7, 3)), ((6, 4), (7, 4)), 2): 407, (((6, 3), (7, 3)), ((6, 2), (7, 2)), 2): 408, (((6, 3), (7, 3)), ((7, 2), (8, 2)), 2): 409, (((7, 3), (8, 3)), ((6, 3), (7, 3)), 2): 410, (((7, 3), (8, 3)), ((6, 4), (7, 4)), 2): 411, (((7, 3), (8, 3)), ((7, 4), (8, 4)), 2): 412, (((7, 3), (8, 3)), ((7, 2), (8, 2)), 2): 413, (((0, 0), (1, 1)), ((2, 2), (1, 1)), 2): 414, (((0, 0), (1, 1)), ((0, 1), (1, 2)), 2): 415, (((0, 0), (1, 1)), ((1, 0), (2, 1)), 2): 416, (((1, 1), (2, 2)), ((0, 0), (1, 1)), 2): 417, (((1, 1), (2, 2)), ((3, 3), (2, 2)), 2): 418, (((1, 1), (2, 2)), ((0, 1), (1, 2)), 2): 419, (((1, 1), (2, 2)), ((1, 2), (2, 3)), 2): 420, (((1, 1), (2, 2)), ((1, 0), (2, 1)), 2): 421, (((1, 1), (2, 2)), ((2, 1), (3, 2)), 2): 422, (((2, 2), (3, 3)), ((1, 1), (2, 2)), 2): 423, (((2, 2), (3, 3)), ((4, 4), (3, 3)), 2): 424, (((2, 2), (3, 3)), ((1, 2), (2, 3)), 2): 425, (((2, 2), (3, 3)), ((2, 3), (3, 4)), 2): 426, (((2, 2), (3, 3)), ((2, 1), (3, 2)), 2): 427, (((2, 2), (3, 3)), ((3, 2), (4, 3)), 2): 428, (((3, 3), (4, 4)), ((2, 2), (3, 3)), 2): 429, (((3, 3), (4, 4)), ((5, 4), (4, 4)), 2): 430, (((3, 3), (4, 4)), ((2, 3), (3, 4)), 2): 431, (((3, 3), (4, 4)), ((3, 4), (4, 5)), 2): 432, (((3, 3), (4, 4)), ((3, 2), (4, 3)), 2): 433, (((3, 3), (4, 4)), ((4, 3), (5, 3)), 2): 434, (((4, 4), (5, 4)), ((3, 3), (4, 4)), 2): 435, (((4, 4), (5, 4)), ((6, 4), (5, 4)), 2): 436, (((4, 4), (5, 4)), ((3, 4), (4, 5)), 2): 437, (((4, 4), (5, 4)), ((4, 5), (5, 5)), 2): 438, (((4, 4), (5, 4)), ((4, 3), (5, 3)), 2): 439, (((4, 4), (5, 4)), ((5, 3), (6, 3)), 2): 440, (((5, 4), (6, 4)), ((4, 4), (5, 4)), 2): 441, (((5, 4), (6, 4)), ((7, 4), (6, 4)), 2): 442, (((5, 4), (6, 4)), ((4, 5), (5, 5)), 2): 443, (((5, 4), (6, 4)), ((5, 5), (6, 5)), 2): 444, (((5, 4), (6, 4)), ((5, 3), (6, 3)), 2): 445, (((5, 4), (6, 4)), ((6, 3), (7, 3)), 2): 446, (((6, 4), (7, 4)), ((5, 4), (6, 4)), 2): 447, (((6, 4), (7, 4)), ((8, 4), (7, 4)), 2): 448, (((6, 4), (7, 4)), ((5, 5), (6, 5)), 2): 449, (((6, 4), (7, 4)), ((6, 5), (7, 5)), 2): 450, (((6, 4), (7, 4)), ((6, 3), (7, 3)), 2): 451, (((6, 4), (7, 4)), ((7, 3), (8, 3)), 2): 452, (((7, 4), (8, 4)), ((6, 4), (7, 4)), 2): 453, (((7, 4), (8, 4)), ((6, 5), (7, 5)), 2): 454, (((7, 4), (8, 4)), ((7, 3), (8, 3)), 2): 455, (((0, 1), (1, 2)), ((2, 3), (1, 2)), 2): 456, (((0, 1), (1, 2)), ((0, 2), (1, 3)), 2): 457, (((0, 1), (1, 2)), ((0, 0), (1, 1)), 2): 458, (((0, 1), (1, 2)), ((1, 1), (2, 2)), 2): 459, (((1, 2), (2, 3)), ((0, 1), (1, 2)), 2): 460, (((1, 2), (2, 3)), ((3, 4), (2, 3)), 2): 461, (((1, 2), (2, 3)), ((0, 2), (1, 3)), 2): 462, (((1, 2), (2, 3)), ((1, 3), (2, 4)), 2): 463, (((1, 2), (2, 3)), ((1, 1), (2, 2)), 2): 464, (((1, 2), (2, 3)), ((2, 2), (3, 3)), 2): 465, (((2, 3), (3, 4)), ((1, 2), (2, 3)), 2): 466, (((2, 3), (3, 4)), ((4, 5), (3, 4)), 2): 467, (((2, 3), (3, 4)), ((1, 3), (2, 4)), 2): 468, (((2, 3), (3, 4)), ((2, 4), (3, 5)), 2): 469, (((2, 3), (3, 4)), ((2, 2), (3, 3)), 2): 470, (((2, 3), (3, 4)), ((3, 3), (4, 4)), 2): 471, (((3, 4), (4, 5)), ((2, 3), (3, 4)), 2): 472, (((3, 4), (4, 5)), ((5, 5), (4, 5)), 2): 473, (((3, 4), (4, 5)), ((2, 4), (3, 5)), 2): 474, (((3, 4), (4, 5)), ((3, 5), (4, 6)), 2): 475, (((3, 4), (4, 5)), ((3, 3), (4, 4)), 2): 476, (((3, 4), (4, 5)), ((4, 4), (5, 4)), 2): 477, (((4, 5), (5, 5)), ((3, 4), (4, 5)), 2): 478, (((4, 5), (5, 5)), ((6, 5), (5, 5)), 2): 479, (((4, 5), (5, 5)), ((3, 5), (4, 6)), 2): 480, (((4, 5), (5, 5)), ((4, 6), (5, 6)), 2): 481, (((4, 5), (5, 5)), ((4, 4), (5, 4)), 2): 482, (((4, 5), (5, 5)), ((5, 4), (6, 4)), 2): 483, (((5, 5), (6, 5)), ((4, 5), (5, 5)), 2): 484, (((5, 5), (6, 5)), ((7, 5), (6, 5)), 2): 485, (((5, 5), (6, 5)), ((4, 6), (5, 6)), 2): 486, (((5, 5), (6, 5)), ((5, 6), (6, 6)), 2): 487, (((5, 5), (6, 5)), ((5, 4), (6, 4)), 2): 488, (((5, 5), (6, 5)), ((6, 4), (7, 4)), 2): 489, (((6, 5), (7, 5)), ((5, 5), (6, 5)), 2): 490, (((6, 5), (7, 5)), ((5, 6), (6, 6)), 2): 491, (((6, 5), (7, 5)), ((6, 4), (7, 4)), 2): 492, (((6, 5), (7, 5)), ((7, 4), (8, 4)), 2): 493, (((0, 2), (1, 3)), ((2, 4), (1, 3)), 2): 494, (((0, 2), (1, 3)), ((0, 3), (1, 4)), 2): 495, (((0, 2), (1, 3)), ((0, 1), (1, 2)), 2): 496, (((0, 2), (1, 3)), ((1, 2), (2, 3)), 2): 497, (((1, 3), (2, 4)), ((0, 2), (1, 3)), 2): 498, (((1, 3), (2, 4)), ((3, 5), (2, 4)), 2): 499, (((1, 3), (2, 4)), ((0, 3), (1, 4)), 2): 500, (((1, 3), (2, 4)), ((1, 4), (2, 5)), 2): 501, (((1, 3), (2, 4)), ((1, 2), (2, 3)), 2): 502, (((1, 3), (2, 4)), ((2, 3), (3, 4)), 2): 503, (((2, 4), (3, 5)), ((1, 3), (2, 4)), 2): 504, (((2, 4), (3, 5)), ((4, 6), (3, 5)), 2): 505, (((2, 4), (3, 5)), ((1, 4), (2, 5)), 2): 506, (((2, 4), (3, 5)), ((2, 5), (3, 6)), 2): 507, (((2, 4), (3, 5)), ((2, 3), (3, 4)), 2): 508, (((2, 4), (3, 5)), ((3, 4), (4, 5)), 2): 509, (((3, 5), (4, 6)), ((2, 4), (3, 5)), 2): 510, (((3, 5), (4, 6)), ((5, 6), (4, 6)), 2): 511, (((3, 5), (4, 6)), ((2, 5), (3, 6)), 2): 512, (((3, 5), (4, 6)), ((3, 6), (4, 7)), 2): 513, (((3, 5), (4, 6)), ((3, 4), (4, 5)), 2): 514, (((3, 5), (4, 6)), ((4, 5), (5, 5)), 2): 515, (((4, 6), (5, 6)), ((3, 5), (4, 6)), 2): 516, (((4, 6), (5, 6)), ((6, 6), (5, 6)), 2): 517, (((4, 6), (5, 6)), ((3, 6), (4, 7)), 2): 518, (((4, 6), (5, 6)), ((4, 7), (5, 7)), 2): 519, (((4, 6), (5, 6)), ((4, 5), (5, 5)), 2): 520, (((4, 6), (5, 6)), ((5, 5), (6, 5)), 2): 521, (((5, 6), (6, 6)), ((4, 6), (5, 6)), 2): 522, (((5, 6), (6, 6)), ((4, 7), (5, 7)), 2): 523, (((5, 6), (6, 6)), ((5, 5), (6, 5)), 2): 524, (((5, 6), (6, 6)), ((6, 5), (7, 5)), 2): 525, (((0, 3), (1, 4)), ((2, 5), (1, 4)), 2): 526, (((0, 3), (1, 4)), ((0, 4), (1, 5)), 2): 527, (((0, 3), (1, 4)), ((0, 2), (1, 3)), 2): 528, (((0, 3), (1, 4)), ((1, 3), (2, 4)), 2): 529, (((1, 4), (2, 5)), ((0, 3), (1, 4)), 2): 530, (((1, 4), (2, 5)), ((3, 6), (2, 5)), 2): 531, (((1, 4), (2, 5)), ((0, 4), (1, 5)), 2): 532, (((1, 4), (2, 5)), ((1, 5), (2, 6)), 2): 533, (((1, 4), (2, 5)), ((1, 3), (2, 4)), 2): 534, (((1, 4), (2, 5)), ((2, 4), (3, 5)), 2): 535, (((2, 5), (3, 6)), ((1, 4), (2, 5)), 2): 536, (((2, 5), (3, 6)), ((4, 7), (3, 6)), 2): 537, (((2, 5), (3, 6)), ((1, 5), (2, 6)), 2): 538, (((2, 5), (3, 6)), ((2, 6), (3, 7)), 2): 539, (((2, 5), (3, 6)), ((2, 4), (3, 5)), 2): 540, (((2, 5), (3, 6)), ((3, 5), (4, 6)), 2): 541, (((3, 6), (4, 7)), ((2, 5), (3, 6)), 2): 542, (((3, 6), (4, 7)), ((5, 7), (4, 7)), 2): 543, (((3, 6), (4, 7)), ((2, 6), (3, 7)), 2): 544, (((3, 6), (4, 7)), ((3, 7), (4, 8)), 2): 545, (((3, 6), (4, 7)), ((3, 5), (4, 6)), 2): 546, (((3, 6), (4, 7)), ((4, 6), (5, 6)), 2): 547, (((4, 7), (5, 7)), ((3, 6), (4, 7)), 2): 548, (((4, 7), (5, 7)), ((3, 7), (4, 8)), 2): 549, (((4, 7), (5, 7)), ((4, 6), (5, 6)), 2): 550, (((4, 7), (5, 7)), ((5, 6), (6, 6)), 2): 551, (((0, 4), (1, 5)), ((2, 6), (1, 5)), 2): 552, (((1, 5), (2, 6)), ((0, 4), (1, 5)), 2): 553, (((1, 5), (2, 6)), ((3, 7), (2, 6)), 2): 554, (((2, 6), (3, 7)), ((1, 5), (2, 6)), 2): 555, (((2, 6), (3, 7)), ((4, 8), (3, 7)), 2): 556, (((3, 7), (4, 8)), ((2, 6), (3, 7)), 2): 557, (((0, 0), (0, 1)), ((0, 2), (0, 1)), 2): 558, (((0, 1), (0, 2)), ((0, 0), (0, 1)), 2): 559, (((0, 1), (0, 2)), ((0, 3), (0, 2)), 2): 560, (((0, 2), (0, 3)), ((0, 1), (0, 2)), 2): 561, (((0, 2), (0, 3)), ((0, 4), (0, 3)), 2): 562, (((0, 3), (0, 4)), ((0, 2), (0, 3)), 2): 563, (((1, 0), (1, 1)), ((1, 2), (1, 1)), 2): 564, (((1, 0), (1, 1)), ((0, 0), (0, 1)), 2): 565, (((1, 0), (1, 1)), ((2, 0), (2, 1)), 2): 566, (((1, 0), (1, 1)), ((2, 1), (2, 2)), 2): 567, (((1, 1), (1, 2)), ((1, 0), (1, 1)), 2): 568, (((1, 1), (1, 2)), ((1, 3), (1, 2)), 2): 569, (((1, 1), (1, 2)), ((2, 1), (2, 2)), 2): 570, (((1, 1), (1, 2)), ((2, 2), (2, 3)), 2): 571, (((1, 1), (1, 2)), ((0, 0), (0, 1)), 2): 572, (((1, 1), (1, 2)), ((0, 1), (0, 2)), 2): 573, (((1, 2), (1, 3)), ((1, 1), (1, 2)), 2): 574, (((1, 2), (1, 3)), ((1, 4), (1, 3)), 2): 575, (((1, 2), (1, 3)), ((2, 2), (2, 3)), 2): 576, (((1, 2), (1, 3)), ((2, 3), (2, 4)), 2): 577, (((1, 2), (1, 3)), ((0, 1), (0, 2)), 2): 578, (((1, 2), (1, 3)), ((0, 2), (0, 3)), 2): 579, (((1, 3), (1, 4)), ((1, 2), (1, 3)), 2): 580, (((1, 3), (1, 4)), ((1, 5), (1, 4)), 2): 581, (((1, 3), (1, 4)), ((2, 3), (2, 4)), 2): 582, (((1, 3), (1, 4)), ((2, 4), (2, 5)), 2): 583, (((1, 3), (1, 4)), ((0, 2), (0, 3)), 2): 584, (((1, 3), (1, 4)), ((0, 3), (0, 4)), 2): 585, (((1, 4), (1, 5)), ((1, 3), (1, 4)), 2): 586, (((1, 4), (1, 5)), ((2, 4), (2, 5)), 2): 587, (((1, 4), (1, 5)), ((2, 5), (2, 6)), 2): 588, (((1, 4), (1, 5)), ((0, 3), (0, 4)), 2): 589, (((2, 0), (2, 1)), ((2, 2), (2, 1)), 2): 590, (((2, 0), (2, 1)), ((1, 0), (1, 1)), 2): 591, (((2, 0), (2, 1)), ((3, 0), (3, 1)), 2): 592, (((2, 0), (2, 1)), ((3, 1), (3, 2)), 2): 593, (((2, 1), (2, 2)), ((2, 0), (2, 1)), 2): 594, (((2, 1), (2, 2)), ((2, 3), (2, 2)), 2): 595, (((2, 1), (2, 2)), ((3, 1), (3, 2)), 2): 596, (((2, 1), (2, 2)), ((3, 2), (3, 3)), 2): 597, (((2, 1), (2, 2)), ((1, 0), (1, 1)), 2): 598, (((2, 1), (2, 2)), ((1, 1), (1, 2)), 2): 599, (((2, 2), (2, 3)), ((2, 1), (2, 2)), 2): 600, (((2, 2), (2, 3)), ((2, 4), (2, 3)), 2): 601, (((2, 2), (2, 3)), ((3, 2), (3, 3)), 2): 602, (((2, 2), (2, 3)), ((3, 3), (3, 4)), 2): 603, (((2, 2), (2, 3)), ((1, 1), (1, 2)), 2): 604, (((2, 2), (2, 3)), ((1, 2), (1, 3)), 2): 605, (((2, 3), (2, 4)), ((2, 2), (2, 3)), 2): 606, (((2, 3), (2, 4)), ((2, 5), (2, 4)), 2): 607, (((2, 3), (2, 4)), ((3, 3), (3, 4)), 2): 608, (((2, 3), (2, 4)), ((3, 4), (3, 5)), 2): 609, (((2, 3), (2, 4)), ((1, 2), (1, 3)), 2): 610, (((2, 3), (2, 4)), ((1, 3), (1, 4)), 2): 611, (((2, 4), (2, 5)), ((2, 3), (2, 4)), 2): 612, (((2, 4), (2, 5)), ((2, 6), (2, 5)), 2): 613, (((2, 4), (2, 5)), ((3, 4), (3, 5)), 2): 614, (((2, 4), (2, 5)), ((3, 5), (3, 6)), 2): 615, (((2, 4), (2, 5)), ((1, 3), (1, 4)), 2): 616, (((2, 4), (2, 5)), ((1, 4), (1, 5)), 2): 617, (((2, 5), (2, 6)), ((2, 4), (2, 5)), 2): 618, (((2, 5), (2, 6)), ((3, 5), (3, 6)), 2): 619, (((2, 5), (2, 6)), ((3, 6), (3, 7)), 2): 620, (((2, 5), (2, 6)), ((1, 4), (1, 5)), 2): 621, (((3, 0), (3, 1)), ((3, 2), (3, 1)), 2): 622, (((3, 0), (3, 1)), ((2, 0), (2, 1)), 2): 623, (((3, 0), (3, 1)), ((4, 0), (4, 1)), 2): 624, (((3, 0), (3, 1)), ((4, 1), (4, 2)), 2): 625, (((3, 1), (3, 2)), ((3, 0), (3, 1)), 2): 626, (((3, 1), (3, 2)), ((3, 3), (3, 2)), 2): 627, (((3, 1), (3, 2)), ((4, 1), (4, 2)), 2): 628, (((3, 1), (3, 2)), ((4, 2), (4, 3)), 2): 629, (((3, 1), (3, 2)), ((2, 0), (2, 1)), 2): 630, (((3, 1), (3, 2)), ((2, 1), (2, 2)), 2): 631, (((3, 2), (3, 3)), ((3, 1), (3, 2)), 2): 632, (((3, 2), (3, 3)), ((3, 4), (3, 3)), 2): 633, (((3, 2), (3, 3)), ((4, 2), (4, 3)), 2): 634, (((3, 2), (3, 3)), ((4, 3), (4, 4)), 2): 635, (((3, 2), (3, 3)), ((2, 1), (2, 2)), 2): 636, (((3, 2), (3, 3)), ((2, 2), (2, 3)), 2): 637, (((3, 3), (3, 4)), ((3, 2), (3, 3)), 2): 638, (((3, 3), (3, 4)), ((3, 5), (3, 4)), 2): 639, (((3, 3), (3, 4)), ((4, 3), (4, 4)), 2): 640, (((3, 3), (3, 4)), ((4, 4), (4, 5)), 2): 641, (((3, 3), (3, 4)), ((2, 2), (2, 3)), 2): 642, (((3, 3), (3, 4)), ((2, 3), (2, 4)), 2): 643, (((3, 4), (3, 5)), ((3, 3), (3, 4)), 2): 644, (((3, 4), (3, 5)), ((3, 6), (3, 5)), 2): 645, (((3, 4), (3, 5)), ((4, 4), (4, 5)), 2): 646, (((3, 4), (3, 5)), ((4, 5), (4, 6)), 2): 647, (((3, 4), (3, 5)), ((2, 3), (2, 4)), 2): 648, (((3, 4), (3, 5)), ((2, 4), (2, 5)), 2): 649, (((3, 5), (3, 6)), ((3, 4), (3, 5)), 2): 650, (((3, 5), (3, 6)), ((3, 7), (3, 6)), 2): 651, (((3, 5), (3, 6)), ((4, 5), (4, 6)), 2): 652, (((3, 5), (3, 6)), ((4, 6), (4, 7)), 2): 653, (((3, 5), (3, 6)), ((2, 4), (2, 5)), 2): 654, (((3, 5), (3, 6)), ((2, 5), (2, 6)), 2): 655, (((3, 6), (3, 7)), ((3, 5), (3, 6)), 2): 656, (((3, 6), (3, 7)), ((4, 6), (4, 7)), 2): 657, (((3, 6), (3, 7)), ((4, 7), (4, 8)), 2): 658, (((3, 6), (3, 7)), ((2, 5), (2, 6)), 2): 659, (((4, 0), (4, 1)), ((4, 2), (4, 1)), 2): 660, (((4, 0), (4, 1)), ((5, 0), (5, 1)), 2): 661, (((4, 0), (4, 1)), ((3, 0), (3, 1)), 2): 662, (((4, 1), (4, 2)), ((4, 0), (4, 1)), 2): 663, (((4, 1), (4, 2)), ((4, 3), (4, 2)), 2): 664, (((4, 1), (4, 2)), ((5, 0), (5, 1)), 2): 665, (((4, 1), (4, 2)), ((5, 1), (5, 2)), 2): 666, (((4, 1), (4, 2)), ((3, 0), (3, 1)), 2): 667, (((4, 1), (4, 2)), ((3, 1), (3, 2)), 2): 668, (((4, 2), (4, 3)), ((4, 1), (4, 2)), 2): 669, (((4, 2), (4, 3)), ((4, 4), (4, 3)), 2): 670, (((4, 2), (4, 3)), ((5, 1), (5, 2)), 2): 671, (((4, 2), (4, 3)), ((5, 2), (5, 3)), 2): 672, (((4, 2), (4, 3)), ((3, 1), (3, 2)), 2): 673, (((4, 2), (4, 3)), ((3, 2), (3, 3)), 2): 674, (((4, 3), (4, 4)), ((4, 2), (4, 3)), 2): 675, (((4, 3), (4, 4)), ((4, 5), (4, 4)), 2): 676, (((4, 3), (4, 4)), ((5, 2), (5, 3)), 2): 677, (((4, 3), (4, 4)), ((5, 3), (5, 4)), 2): 678, (((4, 3), (4, 4)), ((3, 2), (3, 3)), 2): 679, (((4, 3), (4, 4)), ((3, 3), (3, 4)), 2): 680, (((4, 4), (4, 5)), ((4, 3), (4, 4)), 2): 681, (((4, 4), (4, 5)), ((4, 6), (4, 5)), 2): 682, (((4, 4), (4, 5)), ((5, 3), (5, 4)), 2): 683, (((4, 4), (4, 5)), ((5, 4), (5, 5)), 2): 684, (((4, 4), (4, 5)), ((3, 3), (3, 4)), 2): 685, (((4, 4), (4, 5)), ((3, 4), (3, 5)), 2): 686, (((4, 5), (4, 6)), ((4, 4), (4, 5)), 2): 687, (((4, 5), (4, 6)), ((4, 7), (4, 6)), 2): 688, (((4, 5), (4, 6)), ((5, 4), (5, 5)), 2): 689, (((4, 5), (4, 6)), ((5, 5), (5, 6)), 2): 690, (((4, 5), (4, 6)), ((3, 4), (3, 5)), 2): 691, (((4, 5), (4, 6)), ((3, 5), (3, 6)), 2): 692, (((4, 6), (4, 7)), ((4, 5), (4, 6)), 2): 693, (((4, 6), (4, 7)), ((4, 8), (4, 7)), 2): 694, (((4, 6), (4, 7)), ((5, 5), (5, 6)), 2): 695, (((4, 6), (4, 7)), ((5, 6), (5, 7)), 2): 696, (((4, 6), (4, 7)), ((3, 5), (3, 6)), 2): 697, (((4, 6), (4, 7)), ((3, 6), (3, 7)), 2): 698, (((4, 7), (4, 8)), ((4, 6), (4, 7)), 2): 699, (((4, 7), (4, 8)), ((5, 6), (5, 7)), 2): 700, (((4, 7), (4, 8)), ((3, 6), (3, 7)), 2): 701, (((5, 0), (5, 1)), ((5, 2), (5, 1)), 2): 702, (((5, 0), (5, 1)), ((6, 0), (6, 1)), 2): 703, (((5, 0), (5, 1)), ((4, 0), (4, 1)), 2): 704, (((5, 0), (5, 1)), ((4, 1), (4, 2)), 2): 705, (((5, 1), (5, 2)), ((5, 0), (5, 1)), 2): 706, (((5, 1), (5, 2)), ((5, 3), (5, 2)), 2): 707, (((5, 1), (5, 2)), ((6, 0), (6, 1)), 2): 708, (((5, 1), (5, 2)), ((6, 1), (6, 2)), 2): 709, (((5, 1), (5, 2)), ((4, 1), (4, 2)), 2): 710, (((5, 1), (5, 2)), ((4, 2), (4, 3)), 2): 711, (((5, 2), (5, 3)), ((5, 1), (5, 2)), 2): 712, (((5, 2), (5, 3)), ((5, 4), (5, 3)), 2): 713, (((5, 2), (5, 3)), ((6, 1), (6, 2)), 2): 714, (((5, 2), (5, 3)), ((6, 2), (6, 3)), 2): 715, (((5, 2), (5, 3)), ((4, 2), (4, 3)), 2): 716, (((5, 2), (5, 3)), ((4, 3), (4, 4)), 2): 717, (((5, 3), (5, 4)), ((5, 2), (5, 3)), 2): 718, (((5, 3), (5, 4)), ((5, 5), (5, 4)), 2): 719, (((5, 3), (5, 4)), ((6, 2), (6, 3)), 2): 720, (((5, 3), (5, 4)), ((6, 3), (6, 4)), 2): 721, (((5, 3), (5, 4)), ((4, 3), (4, 4)), 2): 722, (((5, 3), (5, 4)), ((4, 4), (4, 5)), 2): 723, (((5, 4), (5, 5)), ((5, 3), (5, 4)), 2): 724, (((5, 4), (5, 5)), ((5, 6), (5, 5)), 2): 725, (((5, 4), (5, 5)), ((6, 3), (6, 4)), 2): 726, (((5, 4), (5, 5)), ((6, 4), (6, 5)), 2): 727, (((5, 4), (5, 5)), ((4, 4), (4, 5)), 2): 728, (((5, 4), (5, 5)), ((4, 5), (4, 6)), 2): 729, (((5, 5), (5, 6)), ((5, 4), (5, 5)), 2): 730, (((5, 5), (5, 6)), ((5, 7), (5, 6)), 2): 731, (((5, 5), (5, 6)), ((6, 4), (6, 5)), 2): 732, (((5, 5), (5, 6)), ((6, 5), (6, 6)), 2): 733, (((5, 5), (5, 6)), ((4, 5), (4, 6)), 2): 734, (((5, 5), (5, 6)), ((4, 6), (4, 7)), 2): 735, (((5, 6), (5, 7)), ((5, 5), (5, 6)), 2): 736, (((5, 6), (5, 7)), ((6, 5), (6, 6)), 2): 737, (((5, 6), (5, 7)), ((4, 6), (4, 7)), 2): 738, (((5, 6), (5, 7)), ((4, 7), (4, 8)), 2): 739, (((6, 0), (6, 1)), ((6, 2), (6, 1)), 2): 740, (((6, 0), (6, 1)), ((7, 0), (7, 1)), 2): 741, (((6, 0), (6, 1)), ((5, 0), (5, 1)), 2): 742, (((6, 0), (6, 1)), ((5, 1), (5, 2)), 2): 743, (((6, 1), (6, 2)), ((6, 0), (6, 1)), 2): 744, (((6, 1), (6, 2)), ((6, 3), (6, 2)), 2): 745, (((6, 1), (6, 2)), ((7, 0), (7, 1)), 2): 746, (((6, 1), (6, 2)), ((7, 1), (7, 2)), 2): 747, (((6, 1), (6, 2)), ((5, 1), (5, 2)), 2): 748, (((6, 1), (6, 2)), ((5, 2), (5, 3)), 2): 749, (((6, 2), (6, 3)), ((6, 1), (6, 2)), 2): 750, (((6, 2), (6, 3)), ((6, 4), (6, 3)), 2): 751, (((6, 2), (6, 3)), ((7, 1), (7, 2)), 2): 752, (((6, 2), (6, 3)), ((7, 2), (7, 3)), 2): 753, (((6, 2), (6, 3)), ((5, 2), (5, 3)), 2): 754, (((6, 2), (6, 3)), ((5, 3), (5, 4)), 2): 755, (((6, 3), (6, 4)), ((6, 2), (6, 3)), 2): 756, (((6, 3), (6, 4)), ((6, 5), (6, 4)), 2): 757, (((6, 3), (6, 4)), ((7, 2), (7, 3)), 2): 758, (((6, 3), (6, 4)), ((7, 3), (7, 4)), 2): 759, (((6, 3), (6, 4)), ((5, 3), (5, 4)), 2): 760, (((6, 3), (6, 4)), ((5, 4), (5, 5)), 2): 761, (((6, 4), (6, 5)), ((6, 3), (6, 4)), 2): 762, (((6, 4), (6, 5)), ((6, 6), (6, 5)), 2): 763, (((6, 4), (6, 5)), ((7, 3), (7, 4)), 2): 764, (((6, 4), (6, 5)), ((7, 4), (7, 5)), 2): 765, (((6, 4), (6, 5)), ((5, 4), (5, 5)), 2): 766, (((6, 4), (6, 5)), ((5, 5), (5, 6)), 2): 767, (((6, 5), (6, 6)), ((6, 4), (6, 5)), 2): 768, (((6, 5), (6, 6)), ((7, 4), (7, 5)), 2): 769, (((6, 5), (6, 6)), ((5, 5), (5, 6)), 2): 770, (((6, 5), (6, 6)), ((5, 6), (5, 7)), 2): 771, (((7, 0), (7, 1)), ((7, 2), (7, 1)), 2): 772, (((7, 0), (7, 1)), ((8, 0), (8, 1)), 2): 773, (((7, 0), (7, 1)), ((6, 0), (6, 1)), 2): 774, (((7, 0), (7, 1)), ((6, 1), (6, 2)), 2): 775, (((7, 1), (7, 2)), ((7, 0), (7, 1)), 2): 776, (((7, 1), (7, 2)), ((7, 3), (7, 2)), 2): 777, (((7, 1), (7, 2)), ((8, 0), (8, 1)), 2): 778, (((7, 1), (7, 2)), ((8, 1), (8, 2)), 2): 779, (((7, 1), (7, 2)), ((6, 1), (6, 2)), 2): 780, (((7, 1), (7, 2)), ((6, 2), (6, 3)), 2): 781, (((7, 2), (7, 3)), ((7, 1), (7, 2)), 2): 782, (((7, 2), (7, 3)), ((7, 4), (7, 3)), 2): 783, (((7, 2), (7, 3)), ((8, 1), (8, 2)), 2): 784, (((7, 2), (7, 3)), ((8, 2), (8, 3)), 2): 785, (((7, 2), (7, 3)), ((6, 2), (6, 3)), 2): 786, (((7, 2), (7, 3)), ((6, 3), (6, 4)), 2): 787, (((7, 3), (7, 4)), ((7, 2), (7, 3)), 2): 788, (((7, 3), (7, 4)), ((7, 5), (7, 4)), 2): 789, (((7, 3), (7, 4)), ((8, 2), (8, 3)), 2): 790, (((7, 3), (7, 4)), ((8, 3), (8, 4)), 2): 791, (((7, 3), (7, 4)), ((6, 3), (6, 4)), 2): 792, (((7, 3), (7, 4)), ((6, 4), (6, 5)), 2): 793, (((7, 4), (7, 5)), ((7, 3), (7, 4)), 2): 794, (((7, 4), (7, 5)), ((8, 3), (8, 4)), 2): 795, (((7, 4), (7, 5)), ((6, 4), (6, 5)), 2): 796, (((7, 4), (7, 5)), ((6, 5), (6, 6)), 2): 797, (((8, 0), (8, 1)), ((8, 2), (8, 1)), 2): 798, (((8, 1), (8, 2)), ((8, 0), (8, 1)), 2): 799, (((8, 1), (8, 2)), ((8, 3), (8, 2)), 2): 800, (((8, 2), (8, 3)), ((8, 1), (8, 2)), 2): 801, (((8, 2), (8, 3)), ((8, 4), (8, 3)), 2): 802, (((8, 3), (8, 4)), ((8, 2), (8, 3)), 2): 803, (((4, 8), (5, 7)), ((6, 6), (5, 7)), 2): 804, (((5, 7), (6, 6)), ((4, 8), (5, 7)), 2): 805, (((5, 7), (6, 6)), ((7, 5), (6, 6)), 2): 806, (((6, 6), (7, 5)), ((5, 7), (6, 6)), 2): 807, (((6, 6), (7, 5)), ((8, 4), (7, 5)), 2): 808, (((7, 5), (8, 4)), ((6, 6), (7, 5)), 2): 809, (((3, 7), (4, 7)), ((5, 6), (4, 7)), 2): 810, (((3, 7), (4, 7)), ((2, 6), (3, 6)), 2): 811, (((3, 7), (4, 7)), ((3, 6), (4, 6)), 2): 812, (((3, 7), (4, 7)), ((4, 8), (5, 7)), 2): 813, (((4, 7), (5, 6)), ((3, 7), (4, 7)), 2): 814, (((4, 7), (5, 6)), ((6, 5), (5, 6)), 2): 815, (((4, 7), (5, 6)), ((3, 6), (4, 6)), 2): 816, (((4, 7), (5, 6)), ((4, 6), (5, 5)), 2): 817, (((4, 7), (5, 6)), ((4, 8), (5, 7)), 2): 818, (((4, 7), (5, 6)), ((5, 7), (6, 6)), 2): 819, (((5, 6), (6, 5)), ((4, 7), (5, 6)), 2): 820, (((5, 6), (6, 5)), ((7, 4), (6, 5)), 2): 821, (((5, 6), (6, 5)), ((4, 6), (5, 5)), 2): 822, (((5, 6), (6, 5)), ((5, 5), (6, 4)), 2): 823, (((5, 6), (6, 5)), ((5, 7), (6, 6)), 2): 824, (((5, 6), (6, 5)), ((6, 6), (7, 5)), 2): 825, (((6, 5), (7, 4)), ((5, 6), (6, 5)), 2): 826, (((6, 5), (7, 4)), ((8, 3), (7, 4)), 2): 827, (((6, 5), (7, 4)), ((5, 5), (6, 4)), 2): 828, (((6, 5), (7, 4)), ((6, 4), (7, 3)), 2): 829, (((6, 5), (7, 4)), ((6, 6), (7, 5)), 2): 830, (((6, 5), (7, 4)), ((7, 5), (8, 4)), 2): 831, (((7, 4), (8, 3)), ((6, 5), (7, 4)), 2): 832, (((7, 4), (8, 3)), ((6, 4), (7, 3)), 2): 833, (((7, 4), (8, 3)), ((7, 3), (8, 2)), 2): 834, (((7, 4), (8, 3)), ((7, 5), (8, 4)), 2): 835, (((2, 6), (3, 6)), ((4, 6), (3, 6)), 2): 836, (((2, 6), (3, 6)), ((1, 5), (2, 5)), 2): 837, (((2, 6), (3, 6)), ((2, 5), (3, 5)), 2): 838, (((2, 6), (3, 6)), ((3, 7), (4, 7)), 2): 839, (((3, 6), (4, 6)), ((2, 6), (3, 6)), 2): 840, (((3, 6), (4, 6)), ((5, 5), (4, 6)), 2): 841, (((3, 6), (4, 6)), ((2, 5), (3, 5)), 2): 842, (((3, 6), (4, 6)), ((3, 5), (4, 5)), 2): 843, (((3, 6), (4, 6)), ((3, 7), (4, 7)), 2): 844, (((3, 6), (4, 6)), ((4, 7), (5, 6)), 2): 845, (((4, 6), (5, 5)), ((3, 6), (4, 6)), 2): 846, (((4, 6), (5, 5)), ((6, 4), (5, 5)), 2): 847, (((4, 6), (5, 5)), ((3, 5), (4, 5)), 2): 848, (((4, 6), (5, 5)), ((4, 5), (5, 4)), 2): 849, (((4, 6), (5, 5)), ((4, 7), (5, 6)), 2): 850, (((4, 6), (5, 5)), ((5, 6), (6, 5)), 2): 851, (((5, 5), (6, 4)), ((4, 6), (5, 5)), 2): 852, (((5, 5), (6, 4)), ((7, 3), (6, 4)), 2): 853, (((5, 5), (6, 4)), ((4, 5), (5, 4)), 2): 854, (((5, 5), (6, 4)), ((5, 4), (6, 3)), 2): 855, (((5, 5), (6, 4)), ((5, 6), (6, 5)), 2): 856, (((5, 5), (6, 4)), ((6, 5), (7, 4)), 2): 857, (((6, 4), (7, 3)), ((5, 5), (6, 4)), 2): 858, (((6, 4), (7, 3)), ((8, 2), (7, 3)), 2): 859, (((6, 4), (7, 3)), ((5, 4), (6, 3)), 2): 860, (((6, 4), (7, 3)), ((6, 3), (7, 2)), 2): 861, (((6, 4), (7, 3)), ((6, 5), (7, 4)), 2): 862, (((6, 4), (7, 3)), ((7, 4), (8, 3)), 2): 863, (((7, 3), (8, 2)), ((6, 4), (7, 3)), 2): 864, (((7, 3), (8, 2)), ((6, 3), (7, 2)), 2): 865, (((7, 3), (8, 2)), ((7, 2), (8, 1)), 2): 866, (((7, 3), (8, 2)), ((7, 4), (8, 3)), 2): 867, (((1, 5), (2, 5)), ((3, 5), (2, 5)), 2): 868, (((1, 5), (2, 5)), ((0, 4), (1, 4)), 2): 869, (((1, 5), (2, 5)), ((1, 4), (2, 4)), 2): 870, (((1, 5), (2, 5)), ((2, 6), (3, 6)), 2): 871, (((2, 5), (3, 5)), ((1, 5), (2, 5)), 2): 872, (((2, 5), (3, 5)), ((4, 5), (3, 5)), 2): 873, (((2, 5), (3, 5)), ((1, 4), (2, 4)), 2): 874, (((2, 5), (3, 5)), ((2, 4), (3, 4)), 2): 875, (((2, 5), (3, 5)), ((2, 6), (3, 6)), 2): 876, (((2, 5), (3, 5)), ((3, 6), (4, 6)), 2): 877, (((3, 5), (4, 5)), ((2, 5), (3, 5)), 2): 878, (((3, 5), (4, 5)), ((5, 4), (4, 5)), 2): 879, (((3, 5), (4, 5)), ((2, 4), (3, 4)), 2): 880, (((3, 5), (4, 5)), ((3, 4), (4, 4)), 2): 881, (((3, 5), (4, 5)), ((3, 6), (4, 6)), 2): 882, (((3, 5), (4, 5)), ((4, 6), (5, 5)), 2): 883, (((4, 5), (5, 4)), ((3, 5), (4, 5)), 2): 884, (((4, 5), (5, 4)), ((6, 3), (5, 4)), 2): 885, (((4, 5), (5, 4)), ((3, 4), (4, 4)), 2): 886, (((4, 5), (5, 4)), ((4, 4), (5, 3)), 2): 887, (((4, 5), (5, 4)), ((4, 6), (5, 5)), 2): 888, (((4, 5), (5, 4)), ((5, 5), (6, 4)), 2): 889, (((5, 4), (6, 3)), ((4, 5), (5, 4)), 2): 890, (((5, 4), (6, 3)), ((7, 2), (6, 3)), 2): 891, (((5, 4), (6, 3)), ((4, 4), (5, 3)), 2): 892, (((5, 4), (6, 3)), ((5, 3), (6, 2)), 2): 893, (((5, 4), (6, 3)), ((5, 5), (6, 4)), 2): 894, (((5, 4), (6, 3)), ((6, 4), (7, 3)), 2): 895, (((6, 3), (7, 2)), ((5, 4), (6, 3)), 2): 896, (((6, 3), (7, 2)), ((8, 1), (7, 2)), 2): 897, (((6, 3), (7, 2)), ((5, 3), (6, 2)), 2): 898, (((6, 3), (7, 2)), ((6, 2), (7, 1)), 2): 899, (((6, 3), (7, 2)), ((6, 4), (7, 3)), 2): 900, (((6, 3), (7, 2)), ((7, 3), (8, 2)), 2): 901, (((7, 2), (8, 1)), ((6, 3), (7, 2)), 2): 902, (((7, 2), (8, 1)), ((6, 2), (7, 1)), 2): 903, (((7, 2), (8, 1)), ((7, 1), (8, 0)), 2): 904, (((7, 2), (8, 1)), ((7, 3), (8, 2)), 2): 905, (((0, 4), (1, 4)), ((2, 4), (1, 4)), 2): 906, (((0, 4), (1, 4)), ((0, 3), (1, 3)), 2): 907, (((0, 4), (1, 4)), ((1, 5), (2, 5)), 2): 908, (((1, 4), (2, 4)), ((0, 4), (1, 4)), 2): 909, (((1, 4), (2, 4)), ((3, 4), (2, 4)), 2): 910, (((1, 4), (2, 4)), ((0, 3), (1, 3)), 2): 911, (((1, 4), (2, 4)), ((1, 3), (2, 3)), 2): 912, (((1, 4), (2, 4)), ((1, 5), (2, 5)), 2): 913, (((1, 4), (2, 4)), ((2, 5), (3, 5)), 2): 914, (((2, 4), (3, 4)), ((1, 4), (2, 4)), 2): 915, (((2, 4), (3, 4)), ((4, 4), (3, 4)), 2): 916, (((2, 4), (3, 4)), ((1, 3), (2, 3)), 2): 917, (((2, 4), (3, 4)), ((2, 3), (3, 3)), 2): 918, (((2, 4), (3, 4)), ((2, 5), (3, 5)), 2): 919, (((2, 4), (3, 4)), ((3, 5), (4, 5)), 2): 920, (((3, 4), (4, 4)), ((2, 4), (3, 4)), 2): 921, (((3, 4), (4, 4)), ((5, 3), (4, 4)), 2): 922, (((3, 4), (4, 4)), ((2, 3), (3, 3)), 2): 923, (((3, 4), (4, 4)), ((3, 3), (4, 3)), 2): 924, (((3, 4), (4, 4)), ((3, 5), (4, 5)), 2): 925, (((3, 4), (4, 4)), ((4, 5), (5, 4)), 2): 926, (((4, 4), (5, 3)), ((3, 4), (4, 4)), 2): 927, (((4, 4), (5, 3)), ((6, 2), (5, 3)), 2): 928, (((4, 4), (5, 3)), ((3, 3), (4, 3)), 2): 929, (((4, 4), (5, 3)), ((4, 3), (5, 2)), 2): 930, (((4, 4), (5, 3)), ((4, 5), (5, 4)), 2): 931, (((4, 4), (5, 3)), ((5, 4), (6, 3)), 2): 932, (((5, 3), (6, 2)), ((4, 4), (5, 3)), 2): 933, (((5, 3), (6, 2)), ((7, 1), (6, 2)), 2): 934, (((5, 3), (6, 2)), ((4, 3), (5, 2)), 2): 935, (((5, 3), (6, 2)), ((5, 2), (6, 1)), 2): 936, (((5, 3), (6, 2)), ((5, 4), (6, 3)), 2): 937, (((5, 3), (6, 2)), ((6, 3), (7, 2)), 2): 938, (((6, 2), (7, 1)), ((5, 3), (6, 2)), 2): 939, (((6, 2), (7, 1)), ((8, 0), (7, 1)), 2): 940, (((6, 2), (7, 1)), ((5, 2), (6, 1)), 2): 941, (((6, 2), (7, 1)), ((6, 1), (7, 0)), 2): 942, (((6, 2), (7, 1)), ((6, 3), (7, 2)), 2): 943, (((6, 2), (7, 1)), ((7, 2), (8, 1)), 2): 944, (((7, 1), (8, 0)), ((6, 2), (7, 1)), 2): 945, (((7, 1), (8, 0)), ((6, 1), (7, 0)), 2): 946, (((7, 1), (8, 0)), ((7, 2), (8, 1)), 2): 947, (((0, 3), (1, 3)), ((2, 3), (1, 3)), 2): 948, (((0, 3), (1, 3)), ((0, 2), (1, 2)), 2): 949, (((0, 3), (1, 3)), ((0, 4), (1, 4)), 2): 950, (((0, 3), (1, 3)), ((1, 4), (2, 4)), 2): 951, (((1, 3), (2, 3)), ((0, 3), (1, 3)), 2): 952, (((1, 3), (2, 3)), ((3, 3), (2, 3)), 2): 953, (((1, 3), (2, 3)), ((0, 2), (1, 2)), 2): 954, (((1, 3), (2, 3)), ((1, 2), (2, 2)), 2): 955, (((1, 3), (2, 3)), ((1, 4), (2, 4)), 2): 956, (((1, 3), (2, 3)), ((2, 4), (3, 4)), 2): 957, (((2, 3), (3, 3)), ((1, 3), (2, 3)), 2): 958, (((2, 3), (3, 3)), ((4, 3), (3, 3)), 2): 959, (((2, 3), (3, 3)), ((1, 2), (2, 2)), 2): 960, (((2, 3), (3, 3)), ((2, 2), (3, 2)), 2): 961, (((2, 3), (3, 3)), ((2, 4), (3, 4)), 2): 962, (((2, 3), (3, 3)), ((3, 4), (4, 4)), 2): 963, (((3, 3), (4, 3)), ((2, 3), (3, 3)), 2): 964, (((3, 3), (4, 3)), ((5, 2), (4, 3)), 2): 965, (((3, 3), (4, 3)), ((2, 2), (3, 2)), 2): 966, (((3, 3), (4, 3)), ((3, 2), (4, 2)), 2): 967, (((3, 3), (4, 3)), ((3, 4), (4, 4)), 2): 968, (((3, 3), (4, 3)), ((4, 4), (5, 3)), 2): 969, (((4, 3), (5, 2)), ((3, 3), (4, 3)), 2): 970, (((4, 3), (5, 2)), ((6, 1), (5, 2)), 2): 971, (((4, 3), (5, 2)), ((3, 2), (4, 2)), 2): 972, (((4, 3), (5, 2)), ((4, 2), (5, 1)), 2): 973, (((4, 3), (5, 2)), ((4, 4), (5, 3)), 2): 974, (((4, 3), (5, 2)), ((5, 3), (6, 2)), 2): 975, (((5, 2), (6, 1)), ((4, 3), (5, 2)), 2): 976, (((5, 2), (6, 1)), ((7, 0), (6, 1)), 2): 977, (((5, 2), (6, 1)), ((4, 2), (5, 1)), 2): 978, (((5, 2), (6, 1)), ((5, 1), (6, 0)), 2): 979, (((5, 2), (6, 1)), ((5, 3), (6, 2)), 2): 980, (((5, 2), (6, 1)), ((6, 2), (7, 1)), 2): 981, (((6, 1), (7, 0)), ((5, 2), (6, 1)), 2): 982, (((6, 1), (7, 0)), ((5, 1), (6, 0)), 2): 983, (((6, 1), (7, 0)), ((6, 2), (7, 1)), 2): 984, (((6, 1), (7, 0)), ((7, 1), (8, 0)), 2): 985, (((0, 2), (1, 2)), ((2, 2), (1, 2)), 2): 986, (((0, 2), (1, 2)), ((0, 1), (1, 1)), 2): 987, (((0, 2), (1, 2)), ((0, 3), (1, 3)), 2): 988, (((0, 2), (1, 2)), ((1, 3), (2, 3)), 2): 989, (((1, 2), (2, 2)), ((0, 2), (1, 2)), 2): 990, (((1, 2), (2, 2)), ((3, 2), (2, 2)), 2): 991, (((1, 2), (2, 2)), ((0, 1), (1, 1)), 2): 992, (((1, 2), (2, 2)), ((1, 1), (2, 1)), 2): 993, (((1, 2), (2, 2)), ((1, 3), (2, 3)), 2): 994, (((1, 2), (2, 2)), ((2, 3), (3, 3)), 2): 995, (((2, 2), (3, 2)), ((1, 2), (2, 2)), 2): 996, (((2, 2), (3, 2)), ((4, 2), (3, 2)), 2): 997, (((2, 2), (3, 2)), ((1, 1), (2, 1)), 2): 998, (((2, 2), (3, 2)), ((2, 1), (3, 1)), 2): 999, (((2, 2), (3, 2)), ((2, 3), (3, 3)), 2): 1000, (((2, 2), (3, 2)), ((3, 3), (4, 3)), 2): 1001, (((3, 2), (4, 2)), ((2, 2), (3, 2)), 2): 1002, (((3, 2), (4, 2)), ((5, 1), (4, 2)), 2): 1003, (((3, 2), (4, 2)), ((2, 1), (3, 1)), 2): 1004, (((3, 2), (4, 2)), ((3, 1), (4, 1)), 2): 1005, (((3, 2), (4, 2)), ((3, 3), (4, 3)), 2): 1006, (((3, 2), (4, 2)), ((4, 3), (5, 2)), 2): 1007, (((4, 2), (5, 1)), ((3, 2), (4, 2)), 2): 1008, (((4, 2), (5, 1)), ((6, 0), (5, 1)), 2): 1009, (((4, 2), (5, 1)), ((3, 1), (4, 1)), 2): 1010, (((4, 2), (5, 1)), ((4, 1), (5, 0)), 2): 1011, (((4, 2), (5, 1)), ((4, 3), (5, 2)), 2): 1012, (((4, 2), (5, 1)), ((5, 2), (6, 1)), 2): 1013, (((5, 1), (6, 0)), ((4, 2), (5, 1)), 2): 1014, (((5, 1), (6, 0)), ((4, 1), (5, 0)), 2): 1015, (((5, 1), (6, 0)), ((5, 2), (6, 1)), 2): 1016, (((5, 1), (6, 0)), ((6, 1), (7, 0)), 2): 1017, (((0, 1), (1, 1)), ((2, 1), (1, 1)), 2): 1018, (((0, 1), (1, 1)), ((0, 0), (1, 0)), 2): 1019, (((0, 1), (1, 1)), ((0, 2), (1, 2)), 2): 1020, (((0, 1), (1, 1)), ((1, 2), (2, 2)), 2): 1021, (((1, 1), (2, 1)), ((0, 1), (1, 1)), 2): 1022, (((1, 1), (2, 1)), ((3, 1), (2, 1)), 2): 1023, (((1, 1), (2, 1)), ((0, 0), (1, 0)), 2): 1024, (((1, 1), (2, 1)), ((1, 0), (2, 0)), 2): 1025, (((1, 1), (2, 1)), ((1, 2), (2, 2)), 2): 1026, (((1, 1), (2, 1)), ((2, 2), (3, 2)), 2): 1027, (((2, 1), (3, 1)), ((1, 1), (2, 1)), 2): 1028, (((2, 1), (3, 1)), ((4, 1), (3, 1)), 2): 1029, (((2, 1), (3, 1)), ((1, 0), (2, 0)), 2): 1030, (((2, 1), (3, 1)), ((2, 0), (3, 0)), 2): 1031, (((2, 1), (3, 1)), ((2, 2), (3, 2)), 2): 1032, (((2, 1), (3, 1)), ((3, 2), (4, 2)), 2): 1033, (((3, 1), (4, 1)), ((2, 1), (3, 1)), 2): 1034, (((3, 1), (4, 1)), ((5, 0), (4, 1)), 2): 1035, (((3, 1), (4, 1)), ((2, 0), (3, 0)), 2): 1036, (((3, 1), (4, 1)), ((3, 0), (4, 0)), 2): 1037, (((3, 1), (4, 1)), ((3, 2), (4, 2)), 2): 1038, (((3, 1), (4, 1)), ((4, 2), (5, 1)), 2): 1039, (((4, 1), (5, 0)), ((3, 1), (4, 1)), 2): 1040, (((4, 1), (5, 0)), ((3, 0), (4, 0)), 2): 1041, (((4, 1), (5, 0)), ((4, 2), (5, 1)), 2): 1042, (((4, 1), (5, 0)), ((5, 1), (6, 0)), 2): 1043, (((0, 0), (1, 0)), ((2, 0), (1, 0)), 2): 1044, (((1, 0), (2, 0)), ((0, 0), (1, 0)), 2): 1045, (((1, 0), (2, 0)), ((3, 0), (2, 0)), 2): 1046, (((2, 0), (3, 0)), ((1, 0), (2, 0)), 2): 1047, (((2, 0), (3, 0)), ((4, 0), (3, 0)), 2): 1048, (((3, 0), (4, 0)), ((2, 0), (3, 0)), 2): 1049, (((4, 0), (5, 0), (6, 0)), ((7, 0), (6, 0), (5, 0)), 3): 1050, (((4, 0), (5, 0), (6, 0)), ((3, 0), (4, 1), (5, 1)), 3): 1051, (((4, 0), (5, 0), (6, 0)), ((4, 1), (5, 1), (6, 1)), 3): 1052, (((5, 0), (6, 0), (7, 0)), ((4, 0), (5, 0), (6, 0)), 3): 1053, (((5, 0), (6, 0), (7, 0)), ((8, 0), (7, 0), (6, 0)), 3): 1054, (((5, 0), (6, 0), (7, 0)), ((4, 1), (5, 1), (6, 1)), 3): 1055, (((5, 0), (6, 0), (7, 0)), ((5, 1), (6, 1), (7, 1)), 3): 1056, (((6, 0), (7, 0), (8, 0)), ((5, 0), (6, 0), (7, 0)), 3): 1057, (((6, 0), (7, 0), (8, 0)), ((5, 1), (6, 1), (7, 1)), 3): 1058, (((6, 0), (7, 0), (8, 0)), ((6, 1), (7, 1), (8, 1)), 3): 1059, (((3, 0), (4, 1), (5, 1)), ((6, 1), (5, 1), (4, 1)), 3): 1060, (((3, 0), (4, 1), (5, 1)), ((2, 0), (3, 1), (4, 2)), 3): 1061, (((3, 0), (4, 1), (5, 1)), ((3, 1), (4, 2), (5, 2)), 3): 1062, (((3, 0), (4, 1), (5, 1)), ((4, 0), (5, 0), (6, 0)), 3): 1063, (((4, 1), (5, 1), (6, 1)), ((3, 0), (4, 1), (5, 1)), 3): 1064, (((4, 1), (5, 1), (6, 1)), ((7, 1), (6, 1), (5, 1)), 3): 1065, (((4, 1), (5, 1), (6, 1)), ((3, 1), (4, 2), (5, 2)), 3): 1066, (((4, 1), (5, 1), (6, 1)), ((4, 2), (5, 2), (6, 2)), 3): 1067, (((4, 1), (5, 1), (6, 1)), ((4, 0), (5, 0), (6, 0)), 3): 1068, (((4, 1), (5, 1), (6, 1)), ((5, 0), (6, 0), (7, 0)), 3): 1069, (((5, 1), (6, 1), (7, 1)), ((4, 1), (5, 1), (6, 1)), 3): 1070, (((5, 1), (6, 1), (7, 1)), ((8, 1), (7, 1), (6, 1)), 3): 1071, (((5, 1), (6, 1), (7, 1)), ((4, 2), (5, 2), (6, 2)), 3): 1072, (((5, 1), (6, 1), (7, 1)), ((5, 2), (6, 2), (7, 2)), 3): 1073, (((5, 1), (6, 1), (7, 1)), ((5, 0), (6, 0), (7, 0)), 3): 1074, (((5, 1), (6, 1), (7, 1)), ((6, 0), (7, 0), (8, 0)), 3): 1075, (((6, 1), (7, 1), (8, 1)), ((5, 1), (6, 1), (7, 1)), 3): 1076, (((6, 1), (7, 1), (8, 1)), ((5, 2), (6, 2), (7, 2)), 3): 1077, (((6, 1), (7, 1), (8, 1)), ((6, 2), (7, 2), (8, 2)), 3): 1078, (((6, 1), (7, 1), (8, 1)), ((6, 0), (7, 0), (8, 0)), 3): 1079, (((2, 0), (3, 1), (4, 2)), ((5, 2), (4, 2), (3, 1)), 3): 1080, (((2, 0), (3, 1), (4, 2)), ((1, 0), (2, 1), (3, 2)), 3): 1081, (((2, 0), (3, 1), (4, 2)), ((2, 1), (3, 2), (4, 3)), 3): 1082, (((2, 0), (3, 1), (4, 2)), ((3, 0), (4, 1), (5, 1)), 3): 1083, (((3, 1), (4, 2), (5, 2)), ((2, 0), (3, 1), (4, 2)), 3): 1084, (((3, 1), (4, 2), (5, 2)), ((6, 2), (5, 2), (4, 2)), 3): 1085, (((3, 1), (4, 2), (5, 2)), ((2, 1), (3, 2), (4, 3)), 3): 1086, (((3, 1), (4, 2), (5, 2)), ((3, 2), (4, 3), (5, 3)), 3): 1087, (((3, 1), (4, 2), (5, 2)), ((3, 0), (4, 1), (5, 1)), 3): 1088, (((3, 1), (4, 2), (5, 2)), ((4, 1), (5, 1), (6, 1)), 3): 1089, (((4, 2), (5, 2), (6, 2)), ((3, 1), (4, 2), (5, 2)), 3): 1090, (((4, 2), (5, 2), (6, 2)), ((7, 2), (6, 2), (5, 2)), 3): 1091, (((4, 2), (5, 2), (6, 2)), ((3, 2), (4, 3), (5, 3)), 3): 1092, (((4, 2), (5, 2), (6, 2)), ((4, 3), (5, 3), (6, 3)), 3): 1093, (((4, 2), (5, 2), (6, 2)), ((4, 1), (5, 1), (6, 1)), 3): 1094, (((4, 2), (5, 2), (6, 2)), ((5, 1), (6, 1), (7, 1)), 3): 1095, (((5, 2), (6, 2), (7, 2)), ((4, 2), (5, 2), (6, 2)), 3): 1096, (((5, 2), (6, 2), (7, 2)), ((8, 2), (7, 2), (6, 2)), 3): 1097, (((5, 2), (6, 2), (7, 2)), ((4, 3), (5, 3), (6, 3)), 3): 1098, (((5, 2), (6, 2), (7, 2)), ((5, 3), (6, 3), (7, 3)), 3): 1099, (((5, 2), (6, 2), (7, 2)), ((5, 1), (6, 1), (7, 1)), 3): 1100, (((5, 2), (6, 2), (7, 2)), ((6, 1), (7, 1), (8, 1)), 3): 1101, (((6, 2), (7, 2), (8, 2)), ((5, 2), (6, 2), (7, 2)), 3): 1102, (((6, 2), (7, 2), (8, 2)), ((5, 3), (6, 3), (7, 3)), 3): 1103, (((6, 2), (7, 2), (8, 2)), ((6, 3), (7, 3), (8, 3)), 3): 1104, (((6, 2), (7, 2), (8, 2)), ((6, 1), (7, 1), (8, 1)), 3): 1105, (((1, 0), (2, 1), (3, 2)), ((4, 3), (3, 2), (2, 1)), 3): 1106, (((1, 0), (2, 1), (3, 2)), ((0, 0), (1, 1), (2, 2)), 3): 1107, (((1, 0), (2, 1), (3, 2)), ((1, 1), (2, 2), (3, 3)), 3): 1108, (((1, 0), (2, 1), (3, 2)), ((2, 0), (3, 1), (4, 2)), 3): 1109, (((2, 1), (3, 2), (4, 3)), ((1, 0), (2, 1), (3, 2)), 3): 1110, (((2, 1), (3, 2), (4, 3)), ((5, 3), (4, 3), (3, 2)), 3): 1111, (((2, 1), (3, 2), (4, 3)), ((1, 1), (2, 2), (3, 3)), 3): 1112, (((2, 1), (3, 2), (4, 3)), ((2, 2), (3, 3), (4, 4)), 3): 1113, (((2, 1), (3, 2), (4, 3)), ((2, 0), (3, 1), (4, 2)), 3): 1114, (((2, 1), (3, 2), (4, 3)), ((3, 1), (4, 2), (5, 2)), 3): 1115, (((3, 2), (4, 3), (5, 3)), ((2, 1), (3, 2), (4, 3)), 3): 1116, (((3, 2), (4, 3), (5, 3)), ((6, 3), (5, 3), (4, 3)), 3): 1117, (((3, 2), (4, 3), (5, 3)), ((2, 2), (3, 3), (4, 4)), 3): 1118, (((3, 2), (4, 3), (5, 3)), ((3, 3), (4, 4), (5, 4)), 3): 1119, (((3, 2), (4, 3), (5, 3)), ((3, 1), (4, 2), (5, 2)), 3): 1120, (((3, 2), (4, 3), (5, 3)), ((4, 2), (5, 2), (6, 2)), 3): 1121, (((4, 3), (5, 3), (6, 3)), ((3, 2), (4, 3), (5, 3)), 3): 1122, (((4, 3), (5, 3), (6, 3)), ((7, 3), (6, 3), (5, 3)), 3): 1123, (((4, 3), (5, 3), (6, 3)), ((3, 3), (4, 4), (5, 4)), 3): 1124, (((4, 3), (5, 3), (6, 3)), ((4, 4), (5, 4), (6, 4)), 3): 1125, (((4, 3), (5, 3), (6, 3)), ((4, 2), (5, 2), (6, 2)), 3): 1126, (((4, 3), (5, 3), (6, 3)), ((5, 2), (6, 2), (7, 2)), 3): 1127, (((5, 3), (6, 3), (7, 3)), ((4, 3), (5, 3), (6, 3)), 3): 1128, (((5, 3), (6, 3), (7, 3)), ((8, 3), (7, 3), (6, 3)), 3): 1129, (((5, 3), (6, 3), (7, 3)), ((4, 4), (5, 4), (6, 4)), 3): 1130, (((5, 3), (6, 3), (7, 3)), ((5, 4), (6, 4), (7, 4)), 3): 1131, (((5, 3), (6, 3), (7, 3)), ((5, 2), (6, 2), (7, 2)), 3): 1132, (((5, 3), (6, 3), (7, 3)), ((6, 2), (7, 2), (8, 2)), 3): 1133, (((6, 3), (7, 3), (8, 3)), ((5, 3), (6, 3), (7, 3)), 3): 1134, (((6, 3), (7, 3), (8, 3)), ((5, 4), (6, 4), (7, 4)), 3): 1135, (((6, 3), (7, 3), (8, 3)), ((6, 4), (7, 4), (8, 4)), 3): 1136, (((6, 3), (7, 3), (8, 3)), ((6, 2), (7, 2), (8, 2)), 3): 1137, (((0, 0), (1, 1), (2, 2)), ((3, 3), (2, 2), (1, 1)), 3): 1138, (((0, 0), (1, 1), (2, 2)), ((0, 1), (1, 2), (2, 3)), 3): 1139, (((0, 0), (1, 1), (2, 2)), ((1, 0), (2, 1), (3, 2)), 3): 1140, (((1, 1), (2, 2), (3, 3)), ((0, 0), (1, 1), (2, 2)), 3): 1141, (((1, 1), (2, 2), (3, 3)), ((4, 4), (3, 3), (2, 2)), 3): 1142, (((1, 1), (2, 2), (3, 3)), ((0, 1), (1, 2), (2, 3)), 3): 1143, (((1, 1), (2, 2), (3, 3)), ((1, 2), (2, 3), (3, 4)), 3): 1144, (((1, 1), (2, 2), (3, 3)), ((1, 0), (2, 1), (3, 2)), 3): 1145, (((1, 1), (2, 2), (3, 3)), ((2, 1), (3, 2), (4, 3)), 3): 1146, (((2, 2), (3, 3), (4, 4)), ((1, 1), (2, 2), (3, 3)), 3): 1147, (((2, 2), (3, 3), (4, 4)), ((5, 4), (4, 4), (3, 3)), 3): 1148, (((2, 2), (3, 3), (4, 4)), ((1, 2), (2, 3), (3, 4)), 3): 1149, (((2, 2), (3, 3), (4, 4)), ((2, 3), (3, 4), (4, 5)), 3): 1150, (((2, 2), (3, 3), (4, 4)), ((2, 1), (3, 2), (4, 3)), 3): 1151, (((2, 2), (3, 3), (4, 4)), ((3, 2), (4, 3), (5, 3)), 3): 1152, (((3, 3), (4, 4), (5, 4)), ((2, 2), (3, 3), (4, 4)), 3): 1153, (((3, 3), (4, 4), (5, 4)), ((6, 4), (5, 4), (4, 4)), 3): 1154, (((3, 3), (4, 4), (5, 4)), ((2, 3), (3, 4), (4, 5)), 3): 1155, (((3, 3), (4, 4), (5, 4)), ((3, 4), (4, 5), (5, 5)), 3): 1156, (((3, 3), (4, 4), (5, 4)), ((3, 2), (4, 3), (5, 3)), 3): 1157, (((3, 3), (4, 4), (5, 4)), ((4, 3), (5, 3), (6, 3)), 3): 1158, (((4, 4), (5, 4), (6, 4)), ((3, 3), (4, 4), (5, 4)), 3): 1159, (((4, 4), (5, 4), (6, 4)), ((7, 4), (6, 4), (5, 4)), 3): 1160, (((4, 4), (5, 4), (6, 4)), ((3, 4), (4, 5), (5, 5)), 3): 1161, (((4, 4), (5, 4), (6, 4)), ((4, 5), (5, 5), (6, 5)), 3): 1162, (((4, 4), (5, 4), (6, 4)), ((4, 3), (5, 3), (6, 3)), 3): 1163, (((4, 4), (5, 4), (6, 4)), ((5, 3), (6, 3), (7, 3)), 3): 1164, (((5, 4), (6, 4), (7, 4)), ((4, 4), (5, 4), (6, 4)), 3): 1165, (((5, 4), (6, 4), (7, 4)), ((8, 4), (7, 4), (6, 4)), 3): 1166, (((5, 4), (6, 4), (7, 4)), ((4, 5), (5, 5), (6, 5)), 3): 1167, (((5, 4), (6, 4), (7, 4)), ((5, 5), (6, 5), (7, 5)), 3): 1168, (((5, 4), (6, 4), (7, 4)), ((5, 3), (6, 3), (7, 3)), 3): 1169, (((5, 4), (6, 4), (7, 4)), ((6, 3), (7, 3), (8, 3)), 3): 1170, (((6, 4), (7, 4), (8, 4)), ((5, 4), (6, 4), (7, 4)), 3): 1171, (((6, 4), (7, 4), (8, 4)), ((5, 5), (6, 5), (7, 5)), 3): 1172, (((6, 4), (7, 4), (8, 4)), ((6, 3), (7, 3), (8, 3)), 3): 1173, (((0, 1), (1, 2), (2, 3)), ((3, 4), (2, 3), (1, 2)), 3): 1174, (((0, 1), (1, 2), (2, 3)), ((0, 2), (1, 3), (2, 4)), 3): 1175, (((0, 1), (1, 2), (2, 3)), ((0, 0), (1, 1), (2, 2)), 3): 1176, (((0, 1), (1, 2), (2, 3)), ((1, 1), (2, 2), (3, 3)), 3): 1177, (((1, 2), (2, 3), (3, 4)), ((0, 1), (1, 2), (2, 3)), 3): 1178, (((1, 2), (2, 3), (3, 4)), ((4, 5), (3, 4), (2, 3)), 3): 1179, (((1, 2), (2, 3), (3, 4)), ((0, 2), (1, 3), (2, 4)), 3): 1180, (((1, 2), (2, 3), (3, 4)), ((1, 3), (2, 4), (3, 5)), 3): 1181, (((1, 2), (2, 3), (3, 4)), ((1, 1), (2, 2), (3, 3)), 3): 1182, (((1, 2), (2, 3), (3, 4)), ((2, 2), (3, 3), (4, 4)), 3): 1183, (((2, 3), (3, 4), (4, 5)), ((1, 2), (2, 3), (3, 4)), 3): 1184, (((2, 3), (3, 4), (4, 5)), ((5, 5), (4, 5), (3, 4)), 3): 1185, (((2, 3), (3, 4), (4, 5)), ((1, 3), (2, 4), (3, 5)), 3): 1186, (((2, 3), (3, 4), (4, 5)), ((2, 4), (3, 5), (4, 6)), 3): 1187, (((2, 3), (3, 4), (4, 5)), ((2, 2), (3, 3), (4, 4)), 3): 1188, (((2, 3), (3, 4), (4, 5)), ((3, 3), (4, 4), (5, 4)), 3): 1189, (((3, 4), (4, 5), (5, 5)), ((2, 3), (3, 4), (4, 5)), 3): 1190, (((3, 4), (4, 5), (5, 5)), ((6, 5), (5, 5), (4, 5)), 3): 1191, (((3, 4), (4, 5), (5, 5)), ((2, 4), (3, 5), (4, 6)), 3): 1192, (((3, 4), (4, 5), (5, 5)), ((3, 5), (4, 6), (5, 6)), 3): 1193, (((3, 4), (4, 5), (5, 5)), ((3, 3), (4, 4), (5, 4)), 3): 1194, (((3, 4), (4, 5), (5, 5)), ((4, 4), (5, 4), (6, 4)), 3): 1195, (((4, 5), (5, 5), (6, 5)), ((3, 4), (4, 5), (5, 5)), 3): 1196, (((4, 5), (5, 5), (6, 5)), ((7, 5), (6, 5), (5, 5)), 3): 1197, (((4, 5), (5, 5), (6, 5)), ((3, 5), (4, 6), (5, 6)), 3): 1198, (((4, 5), (5, 5), (6, 5)), ((4, 6), (5, 6), (6, 6)), 3): 1199, (((4, 5), (5, 5), (6, 5)), ((4, 4), (5, 4), (6, 4)), 3): 1200, (((4, 5), (5, 5), (6, 5)), ((5, 4), (6, 4), (7, 4)), 3): 1201, (((5, 5), (6, 5), (7, 5)), ((4, 5), (5, 5), (6, 5)), 3): 1202, (((5, 5), (6, 5), (7, 5)), ((4, 6), (5, 6), (6, 6)), 3): 1203, (((5, 5), (6, 5), (7, 5)), ((5, 4), (6, 4), (7, 4)), 3): 1204, (((5, 5), (6, 5), (7, 5)), ((6, 4), (7, 4), (8, 4)), 3): 1205, (((0, 2), (1, 3), (2, 4)), ((3, 5), (2, 4), (1, 3)), 3): 1206, (((0, 2), (1, 3), (2, 4)), ((0, 3), (1, 4), (2, 5)), 3): 1207, (((0, 2), (1, 3), (2, 4)), ((0, 1), (1, 2), (2, 3)), 3): 1208, (((0, 2), (1, 3), (2, 4)), ((1, 2), (2, 3), (3, 4)), 3): 1209, (((1, 3), (2, 4), (3, 5)), ((0, 2), (1, 3), (2, 4)), 3): 1210, (((1, 3), (2, 4), (3, 5)), ((4, 6), (3, 5), (2, 4)), 3): 1211, (((1, 3), (2, 4), (3, 5)), ((0, 3), (1, 4), (2, 5)), 3): 1212, (((1, 3), (2, 4), (3, 5)), ((1, 4), (2, 5), (3, 6)), 3): 1213, (((1, 3), (2, 4), (3, 5)), ((1, 2), (2, 3), (3, 4)), 3): 1214, (((1, 3), (2, 4), (3, 5)), ((2, 3), (3, 4), (4, 5)), 3): 1215, (((2, 4), (3, 5), (4, 6)), ((1, 3), (2, 4), (3, 5)), 3): 1216, (((2, 4), (3, 5), (4, 6)), ((5, 6), (4, 6), (3, 5)), 3): 1217, (((2, 4), (3, 5), (4, 6)), ((1, 4), (2, 5), (3, 6)), 3): 1218, (((2, 4), (3, 5), (4, 6)), ((2, 5), (3, 6), (4, 7)), 3): 1219, (((2, 4), (3, 5), (4, 6)), ((2, 3), (3, 4), (4, 5)), 3): 1220, (((2, 4), (3, 5), (4, 6)), ((3, 4), (4, 5), (5, 5)), 3): 1221, (((3, 5), (4, 6), (5, 6)), ((2, 4), (3, 5), (4, 6)), 3): 1222, (((3, 5), (4, 6), (5, 6)), ((6, 6), (5, 6), (4, 6)), 3): 1223, (((3, 5), (4, 6), (5, 6)), ((2, 5), (3, 6), (4, 7)), 3): 1224, (((3, 5), (4, 6), (5, 6)), ((3, 6), (4, 7), (5, 7)), 3): 1225, (((3, 5), (4, 6), (5, 6)), ((3, 4), (4, 5), (5, 5)), 3): 1226, (((3, 5), (4, 6), (5, 6)), ((4, 5), (5, 5), (6, 5)), 3): 1227, (((4, 6), (5, 6), (6, 6)), ((3, 5), (4, 6), (5, 6)), 3): 1228, (((4, 6), (5, 6), (6, 6)), ((3, 6), (4, 7), (5, 7)), 3): 1229, (((4, 6), (5, 6), (6, 6)), ((4, 5), (5, 5), (6, 5)), 3): 1230, (((4, 6), (5, 6), (6, 6)), ((5, 5), (6, 5), (7, 5)), 3): 1231, (((0, 3), (1, 4), (2, 5)), ((3, 6), (2, 5), (1, 4)), 3): 1232, (((0, 3), (1, 4), (2, 5)), ((0, 4), (1, 5), (2, 6)), 3): 1233, (((0, 3), (1, 4), (2, 5)), ((0, 2), (1, 3), (2, 4)), 3): 1234, (((0, 3), (1, 4), (2, 5)), ((1, 3), (2, 4), (3, 5)), 3): 1235, (((1, 4), (2, 5), (3, 6)), ((0, 3), (1, 4), (2, 5)), 3): 1236, (((1, 4), (2, 5), (3, 6)), ((4, 7), (3, 6), (2, 5)), 3): 1237, (((1, 4), (2, 5), (3, 6)), ((0, 4), (1, 5), (2, 6)), 3): 1238, (((1, 4), (2, 5), (3, 6)), ((1, 5), (2, 6), (3, 7)), 3): 1239, (((1, 4), (2, 5), (3, 6)), ((1, 3), (2, 4), (3, 5)), 3): 1240, (((1, 4), (2, 5), (3, 6)), ((2, 4), (3, 5), (4, 6)), 3): 1241, (((2, 5), (3, 6), (4, 7)), ((1, 4), (2, 5), (3, 6)), 3): 1242, (((2, 5), (3, 6), (4, 7)), ((5, 7), (4, 7), (3, 6)), 3): 1243, (((2, 5), (3, 6), (4, 7)), ((1, 5), (2, 6), (3, 7)), 3): 1244, (((2, 5), (3, 6), (4, 7)), ((2, 6), (3, 7), (4, 8)), 3): 1245, (((2, 5), (3, 6), (4, 7)), ((2, 4), (3, 5), (4, 6)), 3): 1246, (((2, 5), (3, 6), (4, 7)), ((3, 5), (4, 6), (5, 6)), 3): 1247, (((3, 6), (4, 7), (5, 7)), ((2, 5), (3, 6), (4, 7)), 3): 1248, (((3, 6), (4, 7), (5, 7)), ((2, 6), (3, 7), (4, 8)), 3): 1249, (((3, 6), (4, 7), (5, 7)), ((3, 5), (4, 6), (5, 6)), 3): 1250, (((3, 6), (4, 7), (5, 7)), ((4, 6), (5, 6), (6, 6)), 3): 1251, (((0, 4), (1, 5), (2, 6)), ((3, 7), (2, 6), (1, 5)), 3): 1252, (((0, 4), (1, 5), (2, 6)), ((0, 3), (1, 4), (2, 5)), 3): 1253, (((0, 4), (1, 5), (2, 6)), ((1, 4), (2, 5), (3, 6)), 3): 1254, (((1, 5), (2, 6), (3, 7)), ((0, 4), (1, 5), (2, 6)), 3): 1255, (((1, 5), (2, 6), (3, 7)), ((4, 8), (3, 7), (2, 6)), 3): 1256, (((1, 5), (2, 6), (3, 7)), ((1, 4), (2, 5), (3, 6)), 3): 1257, (((1, 5), (2, 6), (3, 7)), ((2, 5), (3, 6), (4, 7)), 3): 1258, (((2, 6), (3, 7), (4, 8)), ((1, 5), (2, 6), (3, 7)), 3): 1259, (((2, 6), (3, 7), (4, 8)), ((2, 5), (3, 6), (4, 7)), 3): 1260, (((2, 6), (3, 7), (4, 8)), ((3, 6), (4, 7), (5, 7)), 3): 1261, (((0, 0), (0, 1), (0, 2)), ((0, 3), (0, 2), (0, 1)), 3): 1262, (((0, 0), (0, 1), (0, 2)), ((1, 0), (1, 1), (1, 2)), 3): 1263, (((0, 0), (0, 1), (0, 2)), ((1, 1), (1, 2), (1, 3)), 3): 1264, (((0, 1), (0, 2), (0, 3)), ((0, 0), (0, 1), (0, 2)), 3): 1265, (((0, 1), (0, 2), (0, 3)), ((0, 4), (0, 3), (0, 2)), 3): 1266, (((0, 1), (0, 2), (0, 3)), ((1, 1), (1, 2), (1, 3)), 3): 1267, (((0, 1), (0, 2), (0, 3)), ((1, 2), (1, 3), (1, 4)), 3): 1268, (((0, 2), (0, 3), (0, 4)), ((0, 1), (0, 2), (0, 3)), 3): 1269, (((0, 2), (0, 3), (0, 4)), ((1, 2), (1, 3), (1, 4)), 3): 1270, (((0, 2), (0, 3), (0, 4)), ((1, 3), (1, 4), (1, 5)), 3): 1271, (((1, 0), (1, 1), (1, 2)), ((1, 3), (1, 2), (1, 1)), 3): 1272, (((1, 0), (1, 1), (1, 2)), ((2, 0), (2, 1), (2, 2)), 3): 1273, (((1, 0), (1, 1), (1, 2)), ((2, 1), (2, 2), (2, 3)), 3): 1274, (((1, 0), (1, 1), (1, 2)), ((0, 0), (0, 1), (0, 2)), 3): 1275, (((1, 1), (1, 2), (1, 3)), ((1, 0), (1, 1), (1, 2)), 3): 1276, (((1, 1), (1, 2), (1, 3)), ((1, 4), (1, 3), (1, 2)), 3): 1277, (((1, 1), (1, 2), (1, 3)), ((2, 1), (2, 2), (2, 3)), 3): 1278, (((1, 1), (1, 2), (1, 3)), ((2, 2), (2, 3), (2, 4)), 3): 1279, (((1, 1), (1, 2), (1, 3)), ((0, 0), (0, 1), (0, 2)), 3): 1280, (((1, 1), (1, 2), (1, 3)), ((0, 1), (0, 2), (0, 3)), 3): 1281, (((1, 2), (1, 3), (1, 4)), ((1, 1), (1, 2), (1, 3)), 3): 1282, (((1, 2), (1, 3), (1, 4)), ((1, 5), (1, 4), (1, 3)), 3): 1283, (((1, 2), (1, 3), (1, 4)), ((2, 2), (2, 3), (2, 4)), 3): 1284, (((1, 2), (1, 3), (1, 4)), ((2, 3), (2, 4), (2, 5)), 3): 1285, (((1, 2), (1, 3), (1, 4)), ((0, 1), (0, 2), (0, 3)), 3): 1286, (((1, 2), (1, 3), (1, 4)), ((0, 2), (0, 3), (0, 4)), 3): 1287, (((1, 3), (1, 4), (1, 5)), ((1, 2), (1, 3), (1, 4)), 3): 1288, (((1, 3), (1, 4), (1, 5)), ((2, 3), (2, 4), (2, 5)), 3): 1289, (((1, 3), (1, 4), (1, 5)), ((2, 4), (2, 5), (2, 6)), 3): 1290, (((1, 3), (1, 4), (1, 5)), ((0, 2), (0, 3), (0, 4)), 3): 1291, (((2, 0), (2, 1), (2, 2)), ((2, 3), (2, 2), (2, 1)), 3): 1292, (((2, 0), (2, 1), (2, 2)), ((3, 0), (3, 1), (3, 2)), 3): 1293, (((2, 0), (2, 1), (2, 2)), ((3, 1), (3, 2), (3, 3)), 3): 1294, (((2, 0), (2, 1), (2, 2)), ((1, 0), (1, 1), (1, 2)), 3): 1295, (((2, 1), (2, 2), (2, 3)), ((2, 0), (2, 1), (2, 2)), 3): 1296, (((2, 1), (2, 2), (2, 3)), ((2, 4), (2, 3), (2, 2)), 3): 1297, (((2, 1), (2, 2), (2, 3)), ((3, 1), (3, 2), (3, 3)), 3): 1298, (((2, 1), (2, 2), (2, 3)), ((3, 2), (3, 3), (3, 4)), 3): 1299, (((2, 1), (2, 2), (2, 3)), ((1, 0), (1, 1), (1, 2)), 3): 1300, (((2, 1), (2, 2), (2, 3)), ((1, 1), (1, 2), (1, 3)), 3): 1301, (((2, 2), (2, 3), (2, 4)), ((2, 1), (2, 2), (2, 3)), 3): 1302, (((2, 2), (2, 3), (2, 4)), ((2, 5), (2, 4), (2, 3)), 3): 1303, (((2, 2), (2, 3), (2, 4)), ((3, 2), (3, 3), (3, 4)), 3): 1304, (((2, 2), (2, 3), (2, 4)), ((3, 3), (3, 4), (3, 5)), 3): 1305, (((2, 2), (2, 3), (2, 4)), ((1, 1), (1, 2), (1, 3)), 3): 1306, (((2, 2), (2, 3), (2, 4)), ((1, 2), (1, 3), (1, 4)), 3): 1307, (((2, 3), (2, 4), (2, 5)), ((2, 2), (2, 3), (2, 4)), 3): 1308, (((2, 3), (2, 4), (2, 5)), ((2, 6), (2, 5), (2, 4)), 3): 1309, (((2, 3), (2, 4), (2, 5)), ((3, 3), (3, 4), (3, 5)), 3): 1310, (((2, 3), (2, 4), (2, 5)), ((3, 4), (3, 5), (3, 6)), 3): 1311, (((2, 3), (2, 4), (2, 5)), ((1, 2), (1, 3), (1, 4)), 3): 1312, (((2, 3), (2, 4), (2, 5)), ((1, 3), (1, 4), (1, 5)), 3): 1313, (((2, 4), (2, 5), (2, 6)), ((2, 3), (2, 4), (2, 5)), 3): 1314, (((2, 4), (2, 5), (2, 6)), ((3, 4), (3, 5), (3, 6)), 3): 1315, (((2, 4), (2, 5), (2, 6)), ((3, 5), (3, 6), (3, 7)), 3): 1316, (((2, 4), (2, 5), (2, 6)), ((1, 3), (1, 4), (1, 5)), 3): 1317, (((3, 0), (3, 1), (3, 2)), ((3, 3), (3, 2), (3, 1)), 3): 1318, (((3, 0), (3, 1), (3, 2)), ((4, 0), (4, 1), (4, 2)), 3): 1319, (((3, 0), (3, 1), (3, 2)), ((4, 1), (4, 2), (4, 3)), 3): 1320, (((3, 0), (3, 1), (3, 2)), ((2, 0), (2, 1), (2, 2)), 3): 1321, (((3, 1), (3, 2), (3, 3)), ((3, 0), (3, 1), (3, 2)), 3): 1322, (((3, 1), (3, 2), (3, 3)), ((3, 4), (3, 3), (3, 2)), 3): 1323, (((3, 1), (3, 2), (3, 3)), ((4, 1), (4, 2), (4, 3)), 3): 1324, (((3, 1), (3, 2), (3, 3)), ((4, 2), (4, 3), (4, 4)), 3): 1325, (((3, 1), (3, 2), (3, 3)), ((2, 0), (2, 1), (2, 2)), 3): 1326, (((3, 1), (3, 2), (3, 3)), ((2, 1), (2, 2), (2, 3)), 3): 1327, (((3, 2), (3, 3), (3, 4)), ((3, 1), (3, 2), (3, 3)), 3): 1328, (((3, 2), (3, 3), (3, 4)), ((3, 5), (3, 4), (3, 3)), 3): 1329, (((3, 2), (3, 3), (3, 4)), ((4, 2), (4, 3), (4, 4)), 3): 1330, (((3, 2), (3, 3), (3, 4)), ((4, 3), (4, 4), (4, 5)), 3): 1331, (((3, 2), (3, 3), (3, 4)), ((2, 1), (2, 2), (2, 3)), 3): 1332, (((3, 2), (3, 3), (3, 4)), ((2, 2), (2, 3), (2, 4)), 3): 1333, (((3, 3), (3, 4), (3, 5)), ((3, 2), (3, 3), (3, 4)), 3): 1334, (((3, 3), (3, 4), (3, 5)), ((3, 6), (3, 5), (3, 4)), 3): 1335, (((3, 3), (3, 4), (3, 5)), ((4, 3), (4, 4), (4, 5)), 3): 1336, (((3, 3), (3, 4), (3, 5)), ((4, 4), (4, 5), (4, 6)), 3): 1337, (((3, 3), (3, 4), (3, 5)), ((2, 2), (2, 3), (2, 4)), 3): 1338, (((3, 3), (3, 4), (3, 5)), ((2, 3), (2, 4), (2, 5)), 3): 1339, (((3, 4), (3, 5), (3, 6)), ((3, 3), (3, 4), (3, 5)), 3): 1340, (((3, 4), (3, 5), (3, 6)), ((3, 7), (3, 6), (3, 5)), 3): 1341, (((3, 4), (3, 5), (3, 6)), ((4, 4), (4, 5), (4, 6)), 3): 1342, (((3, 4), (3, 5), (3, 6)), ((4, 5), (4, 6), (4, 7)), 3): 1343, (((3, 4), (3, 5), (3, 6)), ((2, 3), (2, 4), (2, 5)), 3): 1344, (((3, 4), (3, 5), (3, 6)), ((2, 4), (2, 5), (2, 6)), 3): 1345, (((3, 5), (3, 6), (3, 7)), ((3, 4), (3, 5), (3, 6)), 3): 1346, (((3, 5), (3, 6), (3, 7)), ((4, 5), (4, 6), (4, 7)), 3): 1347, (((3, 5), (3, 6), (3, 7)), ((4, 6), (4, 7), (4, 8)), 3): 1348, (((3, 5), (3, 6), (3, 7)), ((2, 4), (2, 5), (2, 6)), 3): 1349, (((4, 0), (4, 1), (4, 2)), ((4, 3), (4, 2), (4, 1)), 3): 1350, (((4, 0), (4, 1), (4, 2)), ((5, 0), (5, 1), (5, 2)), 3): 1351, (((4, 0), (4, 1), (4, 2)), ((3, 0), (3, 1), (3, 2)), 3): 1352, (((4, 1), (4, 2), (4, 3)), ((4, 0), (4, 1), (4, 2)), 3): 1353, (((4, 1), (4, 2), (4, 3)), ((4, 4), (4, 3), (4, 2)), 3): 1354, (((4, 1), (4, 2), (4, 3)), ((5, 0), (5, 1), (5, 2)), 3): 1355, (((4, 1), (4, 2), (4, 3)), ((5, 1), (5, 2), (5, 3)), 3): 1356, (((4, 1), (4, 2), (4, 3)), ((3, 0), (3, 1), (3, 2)), 3): 1357, (((4, 1), (4, 2), (4, 3)), ((3, 1), (3, 2), (3, 3)), 3): 1358, (((4, 2), (4, 3), (4, 4)), ((4, 1), (4, 2), (4, 3)), 3): 1359, (((4, 2), (4, 3), (4, 4)), ((4, 5), (4, 4), (4, 3)), 3): 1360, (((4, 2), (4, 3), (4, 4)), ((5, 1), (5, 2), (5, 3)), 3): 1361, (((4, 2), (4, 3), (4, 4)), ((5, 2), (5, 3), (5, 4)), 3): 1362, (((4, 2), (4, 3), (4, 4)), ((3, 1), (3, 2), (3, 3)), 3): 1363, (((4, 2), (4, 3), (4, 4)), ((3, 2), (3, 3), (3, 4)), 3): 1364, (((4, 3), (4, 4), (4, 5)), ((4, 2), (4, 3), (4, 4)), 3): 1365, (((4, 3), (4, 4), (4, 5)), ((4, 6), (4, 5), (4, 4)), 3): 1366, (((4, 3), (4, 4), (4, 5)), ((5, 2), (5, 3), (5, 4)), 3): 1367, (((4, 3), (4, 4), (4, 5)), ((5, 3), (5, 4), (5, 5)), 3): 1368, (((4, 3), (4, 4), (4, 5)), ((3, 2), (3, 3), (3, 4)), 3): 1369, (((4, 3), (4, 4), (4, 5)), ((3, 3), (3, 4), (3, 5)), 3): 1370, (((4, 4), (4, 5), (4, 6)), ((4, 3), (4, 4), (4, 5)), 3): 1371, (((4, 4), (4, 5), (4, 6)), ((4, 7), (4, 6), (4, 5)), 3): 1372, (((4, 4), (4, 5), (4, 6)), ((5, 3), (5, 4), (5, 5)), 3): 1373, (((4, 4), (4, 5), (4, 6)), ((5, 4), (5, 5), (5, 6)), 3): 1374, (((4, 4), (4, 5), (4, 6)), ((3, 3), (3, 4), (3, 5)), 3): 1375, (((4, 4), (4, 5), (4, 6)), ((3, 4), (3, 5), (3, 6)), 3): 1376, (((4, 5), (4, 6), (4, 7)), ((4, 4), (4, 5), (4, 6)), 3): 1377, (((4, 5), (4, 6), (4, 7)), ((4, 8), (4, 7), (4, 6)), 3): 1378, (((4, 5), (4, 6), (4, 7)), ((5, 4), (5, 5), (5, 6)), 3): 1379, (((4, 5), (4, 6), (4, 7)), ((5, 5), (5, 6), (5, 7)), 3): 1380, (((4, 5), (4, 6), (4, 7)), ((3, 4), (3, 5), (3, 6)), 3): 1381, (((4, 5), (4, 6), (4, 7)), ((3, 5), (3, 6), (3, 7)), 3): 1382, (((4, 6), (4, 7), (4, 8)), ((4, 5), (4, 6), (4, 7)), 3): 1383, (((4, 6), (4, 7), (4, 8)), ((5, 5), (5, 6), (5, 7)), 3): 1384, (((4, 6), (4, 7), (4, 8)), ((3, 5), (3, 6), (3, 7)), 3): 1385, (((5, 0), (5, 1), (5, 2)), ((5, 3), (5, 2), (5, 1)), 3): 1386, (((5, 0), (5, 1), (5, 2)), ((6, 0), (6, 1), (6, 2)), 3): 1387, (((5, 0), (5, 1), (5, 2)), ((4, 0), (4, 1), (4, 2)), 3): 1388, (((5, 0), (5, 1), (5, 2)), ((4, 1), (4, 2), (4, 3)), 3): 1389, (((5, 1), (5, 2), (5, 3)), ((5, 0), (5, 1), (5, 2)), 3): 1390, (((5, 1), (5, 2), (5, 3)), ((5, 4), (5, 3), (5, 2)), 3): 1391, (((5, 1), (5, 2), (5, 3)), ((6, 0), (6, 1), (6, 2)), 3): 1392, (((5, 1), (5, 2), (5, 3)), ((6, 1), (6, 2), (6, 3)), 3): 1393, (((5, 1), (5, 2), (5, 3)), ((4, 1), (4, 2), (4, 3)), 3): 1394, (((5, 1), (5, 2), (5, 3)), ((4, 2), (4, 3), (4, 4)), 3): 1395, (((5, 2), (5, 3), (5, 4)), ((5, 1), (5, 2), (5, 3)), 3): 1396, (((5, 2), (5, 3), (5, 4)), ((5, 5), (5, 4), (5, 3)), 3): 1397, (((5, 2), (5, 3), (5, 4)), ((6, 1), (6, 2), (6, 3)), 3): 1398, (((5, 2), (5, 3), (5, 4)), ((6, 2), (6, 3), (6, 4)), 3): 1399, (((5, 2), (5, 3), (5, 4)), ((4, 2), (4, 3), (4, 4)), 3): 1400, (((5, 2), (5, 3), (5, 4)), ((4, 3), (4, 4), (4, 5)), 3): 1401, (((5, 3), (5, 4), (5, 5)), ((5, 2), (5, 3), (5, 4)), 3): 1402, (((5, 3), (5, 4), (5, 5)), ((5, 6), (5, 5), (5, 4)), 3): 1403, (((5, 3), (5, 4), (5, 5)), ((6, 2), (6, 3), (6, 4)), 3): 1404, (((5, 3), (5, 4), (5, 5)), ((6, 3), (6, 4), (6, 5)), 3): 1405, (((5, 3), (5, 4), (5, 5)), ((4, 3), (4, 4), (4, 5)), 3): 1406, (((5, 3), (5, 4), (5, 5)), ((4, 4), (4, 5), (4, 6)), 3): 1407, (((5, 4), (5, 5), (5, 6)), ((5, 3), (5, 4), (5, 5)), 3): 1408, (((5, 4), (5, 5), (5, 6)), ((5, 7), (5, 6), (5, 5)), 3): 1409, (((5, 4), (5, 5), (5, 6)), ((6, 3), (6, 4), (6, 5)), 3): 1410, (((5, 4), (5, 5), (5, 6)), ((6, 4), (6, 5), (6, 6)), 3): 1411, (((5, 4), (5, 5), (5, 6)), ((4, 4), (4, 5), (4, 6)), 3): 1412, (((5, 4), (5, 5), (5, 6)), ((4, 5), (4, 6), (4, 7)), 3): 1413, (((5, 5), (5, 6), (5, 7)), ((5, 4), (5, 5), (5, 6)), 3): 1414, (((5, 5), (5, 6), (5, 7)), ((6, 4), (6, 5), (6, 6)), 3): 1415, (((5, 5), (5, 6), (5, 7)), ((4, 5), (4, 6), (4, 7)), 3): 1416, (((5, 5), (5, 6), (5, 7)), ((4, 6), (4, 7), (4, 8)), 3): 1417, (((6, 0), (6, 1), (6, 2)), ((6, 3), (6, 2), (6, 1)), 3): 1418, (((6, 0), (6, 1), (6, 2)), ((7, 0), (7, 1), (7, 2)), 3): 1419, (((6, 0), (6, 1), (6, 2)), ((5, 0), (5, 1), (5, 2)), 3): 1420, (((6, 0), (6, 1), (6, 2)), ((5, 1), (5, 2), (5, 3)), 3): 1421, (((6, 1), (6, 2), (6, 3)), ((6, 0), (6, 1), (6, 2)), 3): 1422, (((6, 1), (6, 2), (6, 3)), ((6, 4), (6, 3), (6, 2)), 3): 1423, (((6, 1), (6, 2), (6, 3)), ((7, 0), (7, 1), (7, 2)), 3): 1424, (((6, 1), (6, 2), (6, 3)), ((7, 1), (7, 2), (7, 3)), 3): 1425, (((6, 1), (6, 2), (6, 3)), ((5, 1), (5, 2), (5, 3)), 3): 1426, (((6, 1), (6, 2), (6, 3)), ((5, 2), (5, 3), (5, 4)), 3): 1427, (((6, 2), (6, 3), (6, 4)), ((6, 1), (6, 2), (6, 3)), 3): 1428, (((6, 2), (6, 3), (6, 4)), ((6, 5), (6, 4), (6, 3)), 3): 1429, (((6, 2), (6, 3), (6, 4)), ((7, 1), (7, 2), (7, 3)), 3): 1430, (((6, 2), (6, 3), (6, 4)), ((7, 2), (7, 3), (7, 4)), 3): 1431, (((6, 2), (6, 3), (6, 4)), ((5, 2), (5, 3), (5, 4)), 3): 1432, (((6, 2), (6, 3), (6, 4)), ((5, 3), (5, 4), (5, 5)), 3): 1433, (((6, 3), (6, 4), (6, 5)), ((6, 2), (6, 3), (6, 4)), 3): 1434, (((6, 3), (6, 4), (6, 5)), ((6, 6), (6, 5), (6, 4)), 3): 1435, (((6, 3), (6, 4), (6, 5)), ((7, 2), (7, 3), (7, 4)), 3): 1436, (((6, 3), (6, 4), (6, 5)), ((7, 3), (7, 4), (7, 5)), 3): 1437, (((6, 3), (6, 4), (6, 5)), ((5, 3), (5, 4), (5, 5)), 3): 1438, (((6, 3), (6, 4), (6, 5)), ((5, 4), (5, 5), (5, 6)), 3): 1439, (((6, 4), (6, 5), (6, 6)), ((6, 3), (6, 4), (6, 5)), 3): 1440, (((6, 4), (6, 5), (6, 6)), ((7, 3), (7, 4), (7, 5)), 3): 1441, (((6, 4), (6, 5), (6, 6)), ((5, 4), (5, 5), (5, 6)), 3): 1442, (((6, 4), (6, 5), (6, 6)), ((5, 5), (5, 6), (5, 7)), 3): 1443, (((7, 0), (7, 1), (7, 2)), ((7, 3), (7, 2), (7, 1)), 3): 1444, (((7, 0), (7, 1), (7, 2)), ((8, 0), (8, 1), (8, 2)), 3): 1445, (((7, 0), (7, 1), (7, 2)), ((6, 0), (6, 1), (6, 2)), 3): 1446, (((7, 0), (7, 1), (7, 2)), ((6, 1), (6, 2), (6, 3)), 3): 1447, (((7, 1), (7, 2), (7, 3)), ((7, 0), (7, 1), (7, 2)), 3): 1448, (((7, 1), (7, 2), (7, 3)), ((7, 4), (7, 3), (7, 2)), 3): 1449, (((7, 1), (7, 2), (7, 3)), ((8, 0), (8, 1), (8, 2)), 3): 1450, (((7, 1), (7, 2), (7, 3)), ((8, 1), (8, 2), (8, 3)), 3): 1451, (((7, 1), (7, 2), (7, 3)), ((6, 1), (6, 2), (6, 3)), 3): 1452, (((7, 1), (7, 2), (7, 3)), ((6, 2), (6, 3), (6, 4)), 3): 1453, (((7, 2), (7, 3), (7, 4)), ((7, 1), (7, 2), (7, 3)), 3): 1454, (((7, 2), (7, 3), (7, 4)), ((7, 5), (7, 4), (7, 3)), 3): 1455, (((7, 2), (7, 3), (7, 4)), ((8, 1), (8, 2), (8, 3)), 3): 1456, (((7, 2), (7, 3), (7, 4)), ((8, 2), (8, 3), (8, 4)), 3): 1457, (((7, 2), (7, 3), (7, 4)), ((6, 2), (6, 3), (6, 4)), 3): 1458, (((7, 2), (7, 3), (7, 4)), ((6, 3), (6, 4), (6, 5)), 3): 1459, (((7, 3), (7, 4), (7, 5)), ((7, 2), (7, 3), (7, 4)), 3): 1460, (((7, 3), (7, 4), (7, 5)), ((8, 2), (8, 3), (8, 4)), 3): 1461, (((7, 3), (7, 4), (7, 5)), ((6, 3), (6, 4), (6, 5)), 3): 1462, (((7, 3), (7, 4), (7, 5)), ((6, 4), (6, 5), (6, 6)), 3): 1463, (((8, 0), (8, 1), (8, 2)), ((8, 3), (8, 2), (8, 1)), 3): 1464, (((8, 0), (8, 1), (8, 2)), ((7, 0), (7, 1), (7, 2)), 3): 1465, (((8, 0), (8, 1), (8, 2)), ((7, 1), (7, 2), (7, 3)), 3): 1466, (((8, 1), (8, 2), (8, 3)), ((8, 0), (8, 1), (8, 2)), 3): 1467, (((8, 1), (8, 2), (8, 3)), ((8, 4), (8, 3), (8, 2)), 3): 1468, (((8, 1), (8, 2), (8, 3)), ((7, 1), (7, 2), (7, 3)), 3): 1469, (((8, 1), (8, 2), (8, 3)), ((7, 2), (7, 3), (7, 4)), 3): 1470, (((8, 2), (8, 3), (8, 4)), ((8, 1), (8, 2), (8, 3)), 3): 1471, (((8, 2), (8, 3), (8, 4)), ((7, 2), (7, 3), (7, 4)), 3): 1472, (((8, 2), (8, 3), (8, 4)), ((7, 3), (7, 4), (7, 5)), 3): 1473, (((4, 8), (5, 7), (6, 6)), ((7, 5), (6, 6), (5, 7)), 3): 1474, (((4, 8), (5, 7), (6, 6)), ((3, 7), (4, 7), (5, 6)), 3): 1475, (((4, 8), (5, 7), (6, 6)), ((4, 7), (5, 6), (6, 5)), 3): 1476, (((5, 7), (6, 6), (7, 5)), ((4, 8), (5, 7), (6, 6)), 3): 1477, (((5, 7), (6, 6), (7, 5)), ((8, 4), (7, 5), (6, 6)), 3): 1478, (((5, 7), (6, 6), (7, 5)), ((4, 7), (5, 6), (6, 5)), 3): 1479, (((5, 7), (6, 6), (7, 5)), ((5, 6), (6, 5), (7, 4)), 3): 1480, (((6, 6), (7, 5), (8, 4)), ((5, 7), (6, 6), (7, 5)), 3): 1481, (((6, 6), (7, 5), (8, 4)), ((5, 6), (6, 5), (7, 4)), 3): 1482, (((6, 6), (7, 5), (8, 4)), ((6, 5), (7, 4), (8, 3)), 3): 1483, (((3, 7), (4, 7), (5, 6)), ((6, 5), (5, 6), (4, 7)), 3): 1484, (((3, 7), (4, 7), (5, 6)), ((4, 8), (5, 7), (6, 6)), 3): 1485, (((3, 7), (4, 7), (5, 6)), ((2, 6), (3, 6), (4, 6)), 3): 1486, (((3, 7), (4, 7), (5, 6)), ((3, 6), (4, 6), (5, 5)), 3): 1487, (((4, 7), (5, 6), (6, 5)), ((3, 7), (4, 7), (5, 6)), 3): 1488, (((4, 7), (5, 6), (6, 5)), ((7, 4), (6, 5), (5, 6)), 3): 1489, (((4, 7), (5, 6), (6, 5)), ((4, 8), (5, 7), (6, 6)), 3): 1490, (((4, 7), (5, 6), (6, 5)), ((5, 7), (6, 6), (7, 5)), 3): 1491, (((4, 7), (5, 6), (6, 5)), ((3, 6), (4, 6), (5, 5)), 3): 1492, (((4, 7), (5, 6), (6, 5)), ((4, 6), (5, 5), (6, 4)), 3): 1493, (((5, 6), (6, 5), (7, 4)), ((4, 7), (5, 6), (6, 5)), 3): 1494, (((5, 6), (6, 5), (7, 4)), ((8, 3), (7, 4), (6, 5)), 3): 1495, (((5, 6), (6, 5), (7, 4)), ((5, 7), (6, 6), (7, 5)), 3): 1496, (((5, 6), (6, 5), (7, 4)), ((6, 6), (7, 5), (8, 4)), 3): 1497, (((5, 6), (6, 5), (7, 4)), ((4, 6), (5, 5), (6, 4)), 3): 1498, (((5, 6), (6, 5), (7, 4)), ((5, 5), (6, 4), (7, 3)), 3): 1499, (((6, 5), (7, 4), (8, 3)), ((5, 6), (6, 5), (7, 4)), 3): 1500, (((6, 5), (7, 4), (8, 3)), ((6, 6), (7, 5), (8, 4)), 3): 1501, (((6, 5), (7, 4), (8, 3)), ((5, 5), (6, 4), (7, 3)), 3): 1502, (((6, 5), (7, 4), (8, 3)), ((6, 4), (7, 3), (8, 2)), 3): 1503, (((2, 6), (3, 6), (4, 6)), ((5, 5), (4, 6), (3, 6)), 3): 1504, (((2, 6), (3, 6), (4, 6)), ((3, 7), (4, 7), (5, 6)), 3): 1505, (((2, 6), (3, 6), (4, 6)), ((1, 5), (2, 5), (3, 5)), 3): 1506, (((2, 6), (3, 6), (4, 6)), ((2, 5), (3, 5), (4, 5)), 3): 1507, (((3, 6), (4, 6), (5, 5)), ((2, 6), (3, 6), (4, 6)), 3): 1508, (((3, 6), (4, 6), (5, 5)), ((6, 4), (5, 5), (4, 6)), 3): 1509, (((3, 6), (4, 6), (5, 5)), ((3, 7), (4, 7), (5, 6)), 3): 1510, (((3, 6), (4, 6), (5, 5)), ((4, 7), (5, 6), (6, 5)), 3): 1511, (((3, 6), (4, 6), (5, 5)), ((2, 5), (3, 5), (4, 5)), 3): 1512, (((3, 6), (4, 6), (5, 5)), ((3, 5), (4, 5), (5, 4)), 3): 1513, (((4, 6), (5, 5), (6, 4)), ((3, 6), (4, 6), (5, 5)), 3): 1514, (((4, 6), (5, 5), (6, 4)), ((7, 3), (6, 4), (5, 5)), 3): 1515, (((4, 6), (5, 5), (6, 4)), ((4, 7), (5, 6), (6, 5)), 3): 1516, (((4, 6), (5, 5), (6, 4)), ((5, 6), (6, 5), (7, 4)), 3): 1517, (((4, 6), (5, 5), (6, 4)), ((3, 5), (4, 5), (5, 4)), 3): 1518, (((4, 6), (5, 5), (6, 4)), ((4, 5), (5, 4), (6, 3)), 3): 1519, (((5, 5), (6, 4), (7, 3)), ((4, 6), (5, 5), (6, 4)), 3): 1520, (((5, 5), (6, 4), (7, 3)), ((8, 2), (7, 3), (6, 4)), 3): 1521, (((5, 5), (6, 4), (7, 3)), ((5, 6), (6, 5), (7, 4)), 3): 1522, (((5, 5), (6, 4), (7, 3)), ((6, 5), (7, 4), (8, 3)), 3): 1523, (((5, 5), (6, 4), (7, 3)), ((4, 5), (5, 4), (6, 3)), 3): 1524, (((5, 5), (6, 4), (7, 3)), ((5, 4), (6, 3), (7, 2)), 3): 1525, (((6, 4), (7, 3), (8, 2)), ((5, 5), (6, 4), (7, 3)), 3): 1526, (((6, 4), (7, 3), (8, 2)), ((6, 5), (7, 4), (8, 3)), 3): 1527, (((6, 4), (7, 3), (8, 2)), ((5, 4), (6, 3), (7, 2)), 3): 1528, (((6, 4), (7, 3), (8, 2)), ((6, 3), (7, 2), (8, 1)), 3): 1529, (((1, 5), (2, 5), (3, 5)), ((4, 5), (3, 5), (2, 5)), 3): 1530, (((1, 5), (2, 5), (3, 5)), ((2, 6), (3, 6), (4, 6)), 3): 1531, (((1, 5), (2, 5), (3, 5)), ((0, 4), (1, 4), (2, 4)), 3): 1532, (((1, 5), (2, 5), (3, 5)), ((1, 4), (2, 4), (3, 4)), 3): 1533, (((2, 5), (3, 5), (4, 5)), ((1, 5), (2, 5), (3, 5)), 3): 1534, (((2, 5), (3, 5), (4, 5)), ((5, 4), (4, 5), (3, 5)), 3): 1535, (((2, 5), (3, 5), (4, 5)), ((2, 6), (3, 6), (4, 6)), 3): 1536, (((2, 5), (3, 5), (4, 5)), ((3, 6), (4, 6), (5, 5)), 3): 1537, (((2, 5), (3, 5), (4, 5)), ((1, 4), (2, 4), (3, 4)), 3): 1538, (((2, 5), (3, 5), (4, 5)), ((2, 4), (3, 4), (4, 4)), 3): 1539, (((3, 5), (4, 5), (5, 4)), ((2, 5), (3, 5), (4, 5)), 3): 1540, (((3, 5), (4, 5), (5, 4)), ((6, 3), (5, 4), (4, 5)), 3): 1541, (((3, 5), (4, 5), (5, 4)), ((3, 6), (4, 6), (5, 5)), 3): 1542, (((3, 5), (4, 5), (5, 4)), ((4, 6), (5, 5), (6, 4)), 3): 1543, (((3, 5), (4, 5), (5, 4)), ((2, 4), (3, 4), (4, 4)), 3): 1544, (((3, 5), (4, 5), (5, 4)), ((3, 4), (4, 4), (5, 3)), 3): 1545, (((4, 5), (5, 4), (6, 3)), ((3, 5), (4, 5), (5, 4)), 3): 1546, (((4, 5), (5, 4), (6, 3)), ((7, 2), (6, 3), (5, 4)), 3): 1547, (((4, 5), (5, 4), (6, 3)), ((4, 6), (5, 5), (6, 4)), 3): 1548, (((4, 5), (5, 4), (6, 3)), ((5, 5), (6, 4), (7, 3)), 3): 1549, (((4, 5), (5, 4), (6, 3)), ((3, 4), (4, 4), (5, 3)), 3): 1550, (((4, 5), (5, 4), (6, 3)), ((4, 4), (5, 3), (6, 2)), 3): 1551, (((5, 4), (6, 3), (7, 2)), ((4, 5), (5, 4), (6, 3)), 3): 1552, (((5, 4), (6, 3), (7, 2)), ((8, 1), (7, 2), (6, 3)), 3): 1553, (((5, 4), (6, 3), (7, 2)), ((5, 5), (6, 4), (7, 3)), 3): 1554, (((5, 4), (6, 3), (7, 2)), ((6, 4), (7, 3), (8, 2)), 3): 1555, (((5, 4), (6, 3), (7, 2)), ((4, 4), (5, 3), (6, 2)), 3): 1556, (((5, 4), (6, 3), (7, 2)), ((5, 3), (6, 2), (7, 1)), 3): 1557, (((6, 3), (7, 2), (8, 1)), ((5, 4), (6, 3), (7, 2)), 3): 1558, (((6, 3), (7, 2), (8, 1)), ((6, 4), (7, 3), (8, 2)), 3): 1559, (((6, 3), (7, 2), (8, 1)), ((5, 3), (6, 2), (7, 1)), 3): 1560, (((6, 3), (7, 2), (8, 1)), ((6, 2), (7, 1), (8, 0)), 3): 1561, (((0, 4), (1, 4), (2, 4)), ((3, 4), (2, 4), (1, 4)), 3): 1562, (((0, 4), (1, 4), (2, 4)), ((1, 5), (2, 5), (3, 5)), 3): 1563, (((0, 4), (1, 4), (2, 4)), ((0, 3), (1, 3), (2, 3)), 3): 1564, (((1, 4), (2, 4), (3, 4)), ((0, 4), (1, 4), (2, 4)), 3): 1565, (((1, 4), (2, 4), (3, 4)), ((4, 4), (3, 4), (2, 4)), 3): 1566, (((1, 4), (2, 4), (3, 4)), ((1, 5), (2, 5), (3, 5)), 3): 1567, (((1, 4), (2, 4), (3, 4)), ((2, 5), (3, 5), (4, 5)), 3): 1568, (((1, 4), (2, 4), (3, 4)), ((0, 3), (1, 3), (2, 3)), 3): 1569, (((1, 4), (2, 4), (3, 4)), ((1, 3), (2, 3), (3, 3)), 3): 1570, (((2, 4), (3, 4), (4, 4)), ((1, 4), (2, 4), (3, 4)), 3): 1571, (((2, 4), (3, 4), (4, 4)), ((5, 3), (4, 4), (3, 4)), 3): 1572, (((2, 4), (3, 4), (4, 4)), ((2, 5), (3, 5), (4, 5)), 3): 1573, (((2, 4), (3, 4), (4, 4)), ((3, 5), (4, 5), (5, 4)), 3): 1574, (((2, 4), (3, 4), (4, 4)), ((1, 3), (2, 3), (3, 3)), 3): 1575, (((2, 4), (3, 4), (4, 4)), ((2, 3), (3, 3), (4, 3)), 3): 1576, (((3, 4), (4, 4), (5, 3)), ((2, 4), (3, 4), (4, 4)), 3): 1577, (((3, 4), (4, 4), (5, 3)), ((6, 2), (5, 3), (4, 4)), 3): 1578, (((3, 4), (4, 4), (5, 3)), ((3, 5), (4, 5), (5, 4)), 3): 1579, (((3, 4), (4, 4), (5, 3)), ((4, 5), (5, 4), (6, 3)), 3): 1580, (((3, 4), (4, 4), (5, 3)), ((2, 3), (3, 3), (4, 3)), 3): 1581, (((3, 4), (4, 4), (5, 3)), ((3, 3), (4, 3), (5, 2)), 3): 1582, (((4, 4), (5, 3), (6, 2)), ((3, 4), (4, 4), (5, 3)), 3): 1583, (((4, 4), (5, 3), (6, 2)), ((7, 1), (6, 2), (5, 3)), 3): 1584, (((4, 4), (5, 3), (6, 2)), ((4, 5), (5, 4), (6, 3)), 3): 1585, (((4, 4), (5, 3), (6, 2)), ((5, 4), (6, 3), (7, 2)), 3): 1586, (((4, 4), (5, 3), (6, 2)), ((3, 3), (4, 3), (5, 2)), 3): 1587, (((4, 4), (5, 3), (6, 2)), ((4, 3), (5, 2), (6, 1)), 3): 1588, (((5, 3), (6, 2), (7, 1)), ((4, 4), (5, 3), (6, 2)), 3): 1589, (((5, 3), (6, 2), (7, 1)), ((8, 0), (7, 1), (6, 2)), 3): 1590, (((5, 3), (6, 2), (7, 1)), ((5, 4), (6, 3), (7, 2)), 3): 1591, (((5, 3), (6, 2), (7, 1)), ((6, 3), (7, 2), (8, 1)), 3): 1592, (((5, 3), (6, 2), (7, 1)), ((4, 3), (5, 2), (6, 1)), 3): 1593, (((5, 3), (6, 2), (7, 1)), ((5, 2), (6, 1), (7, 0)), 3): 1594, (((6, 2), (7, 1), (8, 0)), ((5, 3), (6, 2), (7, 1)), 3): 1595, (((6, 2), (7, 1), (8, 0)), ((6, 3), (7, 2), (8, 1)), 3): 1596, (((6, 2), (7, 1), (8, 0)), ((5, 2), (6, 1), (7, 0)), 3): 1597, (((0, 3), (1, 3), (2, 3)), ((3, 3), (2, 3), (1, 3)), 3): 1598, (((0, 3), (1, 3), (2, 3)), ((0, 4), (1, 4), (2, 4)), 3): 1599, (((0, 3), (1, 3), (2, 3)), ((1, 4), (2, 4), (3, 4)), 3): 1600, (((0, 3), (1, 3), (2, 3)), ((0, 2), (1, 2), (2, 2)), 3): 1601, (((1, 3), (2, 3), (3, 3)), ((0, 3), (1, 3), (2, 3)), 3): 1602, (((1, 3), (2, 3), (3, 3)), ((4, 3), (3, 3), (2, 3)), 3): 1603, (((1, 3), (2, 3), (3, 3)), ((1, 4), (2, 4), (3, 4)), 3): 1604, (((1, 3), (2, 3), (3, 3)), ((2, 4), (3, 4), (4, 4)), 3): 1605, (((1, 3), (2, 3), (3, 3)), ((0, 2), (1, 2), (2, 2)), 3): 1606, (((1, 3), (2, 3), (3, 3)), ((1, 2), (2, 2), (3, 2)), 3): 1607, (((2, 3), (3, 3), (4, 3)), ((1, 3), (2, 3), (3, 3)), 3): 1608, (((2, 3), (3, 3), (4, 3)), ((5, 2), (4, 3), (3, 3)), 3): 1609, (((2, 3), (3, 3), (4, 3)), ((2, 4), (3, 4), (4, 4)), 3): 1610, (((2, 3), (3, 3), (4, 3)), ((3, 4), (4, 4), (5, 3)), 3): 1611, (((2, 3), (3, 3), (4, 3)), ((1, 2), (2, 2), (3, 2)), 3): 1612, (((2, 3), (3, 3), (4, 3)), ((2, 2), (3, 2), (4, 2)), 3): 1613, (((3, 3), (4, 3), (5, 2)), ((2, 3), (3, 3), (4, 3)), 3): 1614, (((3, 3), (4, 3), (5, 2)), ((6, 1), (5, 2), (4, 3)), 3): 1615, (((3, 3), (4, 3), (5, 2)), ((3, 4), (4, 4), (5, 3)), 3): 1616, (((3, 3), (4, 3), (5, 2)), ((4, 4), (5, 3), (6, 2)), 3): 1617, (((3, 3), (4, 3), (5, 2)), ((2, 2), (3, 2), (4, 2)), 3): 1618, (((3, 3), (4, 3), (5, 2)), ((3, 2), (4, 2), (5, 1)), 3): 1619, (((4, 3), (5, 2), (6, 1)), ((3, 3), (4, 3), (5, 2)), 3): 1620, (((4, 3), (5, 2), (6, 1)), ((7, 0), (6, 1), (5, 2)), 3): 1621, (((4, 3), (5, 2), (6, 1)), ((4, 4), (5, 3), (6, 2)), 3): 1622, (((4, 3), (5, 2), (6, 1)), ((5, 3), (6, 2), (7, 1)), 3): 1623, (((4, 3), (5, 2), (6, 1)), ((3, 2), (4, 2), (5, 1)), 3): 1624, (((4, 3), (5, 2), (6, 1)), ((4, 2), (5, 1), (6, 0)), 3): 1625, (((5, 2), (6, 1), (7, 0)), ((4, 3), (5, 2), (6, 1)), 3): 1626, (((5, 2), (6, 1), (7, 0)), ((5, 3), (6, 2), (7, 1)), 3): 1627, (((5, 2), (6, 1), (7, 0)), ((6, 2), (7, 1), (8, 0)), 3): 1628, (((5, 2), (6, 1), (7, 0)), ((4, 2), (5, 1), (6, 0)), 3): 1629, (((0, 2), (1, 2), (2, 2)), ((3, 2), (2, 2), (1, 2)), 3): 1630, (((0, 2), (1, 2), (2, 2)), ((0, 3), (1, 3), (2, 3)), 3): 1631, (((0, 2), (1, 2), (2, 2)), ((1, 3), (2, 3), (3, 3)), 3): 1632, (((0, 2), (1, 2), (2, 2)), ((0, 1), (1, 1), (2, 1)), 3): 1633, (((1, 2), (2, 2), (3, 2)), ((0, 2), (1, 2), (2, 2)), 3): 1634, (((1, 2), (2, 2), (3, 2)), ((4, 2), (3, 2), (2, 2)), 3): 1635, (((1, 2), (2, 2), (3, 2)), ((1, 3), (2, 3), (3, 3)), 3): 1636, (((1, 2), (2, 2), (3, 2)), ((2, 3), (3, 3), (4, 3)), 3): 1637, (((1, 2), (2, 2), (3, 2)), ((0, 1), (1, 1), (2, 1)), 3): 1638, (((1, 2), (2, 2), (3, 2)), ((1, 1), (2, 1), (3, 1)), 3): 1639, (((2, 2), (3, 2), (4, 2)), ((1, 2), (2, 2), (3, 2)), 3): 1640, (((2, 2), (3, 2), (4, 2)), ((5, 1), (4, 2), (3, 2)), 3): 1641, (((2, 2), (3, 2), (4, 2)), ((2, 3), (3, 3), (4, 3)), 3): 1642, (((2, 2), (3, 2), (4, 2)), ((3, 3), (4, 3), (5, 2)), 3): 1643, (((2, 2), (3, 2), (4, 2)), ((1, 1), (2, 1), (3, 1)), 3): 1644, (((2, 2), (3, 2), (4, 2)), ((2, 1), (3, 1), (4, 1)), 3): 1645, (((3, 2), (4, 2), (5, 1)), ((2, 2), (3, 2), (4, 2)), 3): 1646, (((3, 2), (4, 2), (5, 1)), ((6, 0), (5, 1), (4, 2)), 3): 1647, (((3, 2), (4, 2), (5, 1)), ((3, 3), (4, 3), (5, 2)), 3): 1648, (((3, 2), (4, 2), (5, 1)), ((4, 3), (5, 2), (6, 1)), 3): 1649, (((3, 2), (4, 2), (5, 1)), ((2, 1), (3, 1), (4, 1)), 3): 1650, (((3, 2), (4, 2), (5, 1)), ((3, 1), (4, 1), (5, 0)), 3): 1651, (((4, 2), (5, 1), (6, 0)), ((3, 2), (4, 2), (5, 1)), 3): 1652, (((4, 2), (5, 1), (6, 0)), ((4, 3), (5, 2), (6, 1)), 3): 1653, (((4, 2), (5, 1), (6, 0)), ((5, 2), (6, 1), (7, 0)), 3): 1654, (((4, 2), (5, 1), (6, 0)), ((3, 1), (4, 1), (5, 0)), 3): 1655, (((0, 1), (1, 1), (2, 1)), ((3, 1), (2, 1), (1, 1)), 3): 1656, (((0, 1), (1, 1), (2, 1)), ((0, 2), (1, 2), (2, 2)), 3): 1657, (((0, 1), (1, 1), (2, 1)), ((1, 2), (2, 2), (3, 2)), 3): 1658, (((0, 1), (1, 1), (2, 1)), ((0, 0), (1, 0), (2, 0)), 3): 1659, (((1, 1), (2, 1), (3, 1)), ((0, 1), (1, 1), (2, 1)), 3): 1660, (((1, 1), (2, 1), (3, 1)), ((4, 1), (3, 1), (2, 1)), 3): 1661, (((1, 1), (2, 1), (3, 1)), ((1, 2), (2, 2), (3, 2)), 3): 1662, (((1, 1), (2, 1), (3, 1)), ((2, 2), (3, 2), (4, 2)), 3): 1663, (((1, 1), (2, 1), (3, 1)), ((0, 0), (1, 0), (2, 0)), 3): 1664, (((1, 1), (2, 1), (3, 1)), ((1, 0), (2, 0), (3, 0)), 3): 1665, (((2, 1), (3, 1), (4, 1)), ((1, 1), (2, 1), (3, 1)), 3): 1666, (((2, 1), (3, 1), (4, 1)), ((5, 0), (4, 1), (3, 1)), 3): 1667, (((2, 1), (3, 1), (4, 1)), ((2, 2), (3, 2), (4, 2)), 3): 1668, (((2, 1), (3, 1), (4, 1)), ((3, 2), (4, 2), (5, 1)), 3): 1669, (((2, 1), (3, 1), (4, 1)), ((1, 0), (2, 0), (3, 0)), 3): 1670, (((2, 1), (3, 1), (4, 1)), ((2, 0), (3, 0), (4, 0)), 3): 1671, (((3, 1), (4, 1), (5, 0)), ((2, 1), (3, 1), (4, 1)), 3): 1672, (((3, 1), (4, 1), (5, 0)), ((3, 2), (4, 2), (5, 1)), 3): 1673, (((3, 1), (4, 1), (5, 0)), ((4, 2), (5, 1), (6, 0)), 3): 1674, (((3, 1), (4, 1), (5, 0)), ((2, 0), (3, 0), (4, 0)), 3): 1675, (((0, 0), (1, 0), (2, 0)), ((3, 0), (2, 0), (1, 0)), 3): 1676, (((0, 0), (1, 0), (2, 0)), ((0, 1), (1, 1), (2, 1)), 3): 1677, (((0, 0), (1, 0), (2, 0)), ((1, 1), (2, 1), (3, 1)), 3): 1678, (((1, 0), (2, 0), (3, 0)), ((0, 0), (1, 0), (2, 0)), 3): 1679, (((1, 0), (2, 0), (3, 0)), ((4, 0), (3, 0), (2, 0)), 3): 1680, (((1, 0), (2, 0), (3, 0)), ((1, 1), (2, 1), (3, 1)), 3): 1681, (((1, 0), (2, 0), (3, 0)), ((2, 1), (3, 1), (4, 1)), 3): 1682, (((2, 0), (3, 0), (4, 0)), ((1, 0), (2, 0), (3, 0)), 3): 1683, (((2, 0), (3, 0), (4, 0)), ((2, 1), (3, 1), (4, 1)), 3): 1684, (((2, 0), (3, 0), (4, 0)), ((3, 1), (4, 1), (5, 0)), 3): 1685}

        return all_actions