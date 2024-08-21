from collections import defaultdict

def is_adjacent(board):
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

# Example board configuration
board = [
    [1,1,1,1,1],
    [1,1,1,1,1,1],
    [0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,-1,-1,-1,0,0],
    [-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1],
]

# Generate the dictionary of adjacent cells
adjacent_cells_dict = is_adjacent(board)



def find_player_balls(board, player):
 
        player_positions = []
        for row_index, row in enumerate(board):
            for col_index, cell in enumerate(row):
                if cell == player:
                    player_positions.append((row_index, col_index))
        return player_positions

player_positions = find_player_balls(board,1)

adjacent_pairs = [
            (ball1, ball2) 
            for i, ball1 in enumerate(player_positions) 
            for ball2 in player_positions[i+1:] 
            if ball2 in adjacent_cells_dict[ball1]
            ]

'''print(adjacent_pairs)'''

def define_straight_lines():

        all_diagonal_left_to_right = [[(4,0),(5,0),(6,0),(7,0),(8,0)],
                              [(3,0),(4,1),(5,1),(6,1),(7,1),(8,1)],
                              [(2,0),(3,1),(4,2),(5,2),(6,2),(7,2),(8,2)],
                              [(1,0),(2,1),(3,2),(4,3),(5,3),(6,3),(7,3),(8,3)],
                              [(0,0), (1,1),(2,2),(3,3),(4,4),(5,4),(6,4),(7,4),(8,4)],
                              [(0,1),(1,2),(2,3),(3,4),(4,5),(5,5),(6,5),(7,5)],
                              [(0,2),(1,3),(2,4),(3,5),(4,6),(5,6),(6,6)],
                              [(0,3),(1,4),(2,5),(3,6),(4,7),(5,7)],
                              [(0,4),(1,5),(2,6),(3,7),(4,8)]]
        
        all_vertical = [[(0,0),(0,1),(0,2),(0,3),(0,4)],
                              [(1,0),(1,1),(1,2),(1,3),(1,4),(1,5)],
                              [(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6)],
                              [(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7)],
                              [(4,0), (4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8)],
                              [(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7)],
                              [(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)],
                              [(7,0),(7,1),(7,2),(7,3),(7,4),(7,5)],
                              [(8,0),(8,1),(8,2),(8,3),(8,4)]]
        
        all_diagonal_right_to_left = [[(4,8),(5,7),(6,6),(7,5),(8,4)],
                              [(3,7),(4,7),(5,6),(6,5),(7,4),(8,3)],
                              [(2,6),(3,6),(4,6),(5,5),(6,4),(7,3),(8,2)],
                              [(1,5),(2,5),(3,5),(4,5),(5,4),(6,3),(7,2),(8,1)],
                              [(0,4), (1,1),(2,4),(3,4),(4,4),(5,3),(6,2),(7,1),(8,0)],
                              [(0,3),(1,3),(2,3),(3,3),(4,3),(5,2),(6,1),(7,0)],
                              [(0,2),(1,2),(2,2),(3,2),(4,2),(5,1),(6,0)],
                              [(0,1),(1,1),(2,1),(3,1),(4,1),(5,0)],
                              [(0,0),(1,0),(2,0),(3,0),(4,0)]]
        
        return all_diagonal_left_to_right + all_vertical + all_diagonal_right_to_left

def create_adjacent_pairs_to_straight_lines(straight_lines):
        adjacent_pairs_dict = {}
        
        # Function to create pairs of adjacent cells
        def create_pairs(line):
            for i in range(len(line) - 1):
                pair = (line[i], line[i + 1])
                # Add both directions of pair to ensure no pair is missed
                adjacent_pairs_dict[pair] = line
                adjacent_pairs_dict[(pair[1], pair[0])] = line

        # Process each line
        for line in straight_lines:
            create_pairs(line)
            
        return adjacent_pairs_dict

adjacent_pairs_all = create_adjacent_pairs_to_straight_lines(define_straight_lines())

def get_cell(row, col):

        if 0 <= row < len(board) and 0 <= col < len(board[row]):
            return board[row][col]
        else:
            return None


def return_parallel_lines(straight_line):

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
                              [(0,4), (1,1),(2,4),(3,4),(4,4),(5,3),(6,2),(7,1),(8,0)],
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
        
def get_legitimate_two_balls_moves(player, player_positions, adjacent_cells=None):
        # Using defaultdict for automatic initialization of missing keys
        legitimate_two_ball_moves = defaultdict(list)

        if adjacent_cells is None:
            adjacent_cells = adjacent_cells_dict

        # Finding adjacent pairs of player's balls
        adjacent_duos_dict = adjacent_pairs_all

        # Find adjacent trios of player's balls
        adjacent_pairs = [
            duo for duo in adjacent_duos_dict
            if all(ball in player_positions for ball in duo)
        ]

        for adjacent_pair in adjacent_pairs:
            straight_line = adjacent_pairs_all[adjacent_pair]

            # Mapping each cell to its index within its line for faster access
            indices = {cell: index for index, cell in enumerate(straight_line)}

            # Define a function to check the move validity in a given direction
            def check_move_direction(cell, direction):
                index = indices[cell]
                next_index = index + direction
                # Check if the next index is within bounds
                if 0 <= next_index < len(straight_line):
                    next_cell = straight_line[next_index]
                    cell_value = get_cell(next_cell[0], next_cell[1])
                    # Check the cell value and act accordingly
                    if cell_value == 0:
                        legitimate_two_ball_moves[adjacent_pair].append((next_cell, cell))
                    elif cell_value != player and cell_value is not None:
                        # If the next cell is not empty and not the player's, check the next one
                        next_next_index = next_index + direction
                        if 0 <= next_next_index < len(straight_line):
                            next_next_cell = straight_line[next_next_index]
                            if get_cell(next_next_cell[0],next_next_cell[1]) == 0:
                                legitimate_two_ball_moves[adjacent_pair].append((next_cell, cell))

            def parallel_moves(pair, straight_line):

                # find the two cells which are adjacent to both the balls in the pair
                adjacent_cells = []
                cell1 = pair[0]
                cell2 = pair[1]
                
                for adj_cell in adjacent_cells_dict[cell1]:
                    if adj_cell in adjacent_cells_dict[cell2]:
                        adjacent_cells.append(adj_cell)
                    
                # now we need to check whether the adjacent cells are empty, and if yes, then also one cell to the right and left since a pair
                # has 4 theorically possible parallel moves
                parralel_lines = return_parallel_lines(straight_line)

                for cell in adjacent_cells:
                    if get_cell(cell[0], cell[1]) == 0:
                        # if the cell is empty, then we need to check the cells to the right and left of it but first check on which
                        # of the parallel lines it lies

                        for line in parralel_lines:
                            if cell in line:
                                parallel_line = line
                                break
                        
                        # now we need to find the index of the cell in the parallel line
                        index = parallel_line.index(cell)
                        # now we need to check the cells to the right and left of the cell
                        if index > 0:
                            if get_cell(parallel_line[index-1][0], parallel_line[index-1][1]) == 0:
                                legitimate_two_ball_moves[pair].append((parallel_line[index-1], cell))

                        if index < len(parallel_line) - 1:
                            
                            if get_cell(parallel_line[index+1][0], parallel_line[index+1][1]) == 0:
                                legitimate_two_ball_moves[pair].append((parallel_line[index+1], cell))

            # Check both directions for each pair
            check_move_direction(adjacent_pair[0], -1)
            check_move_direction(adjacent_pair[1], 1)
            parallel_moves(adjacent_pair, straight_line)

        return dict(legitimate_two_ball_moves)

two_balls_moves= get_legitimate_two_balls_moves(1,player_positions )


        

def get_legitimate_three_balls_moves(player, player_positions):
        # Using defaultdict for automatic initialization of missing keys
        legitimate_three_ball_moves = defaultdict(list)

        # Create the adjacent trios to straight lines dictionary
        adjacent_trios_dict = create_adjacent_trios_to_straight_lines(define_straight_lines())

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
                    cell_value = get_cell(next_cell[0], next_cell[1])
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
                            if get_cell(next_next_cell[0], next_next_cell[1]) == 0:
                                if i == 0:
                                    legitimate_three_ball_moves[trio].append((next_cell, trio[0],trio[1]))
                                else:
                                    legitimate_three_ball_moves[trio].append((next_cell, trio[2],trio[1]))
                            
                        elif get_cell(next_next_cell[0], next_next_cell[1]) != player and get_cell(next_next_cell[0], next_next_cell[1]) != None:
                            next_next_index = next_index + direction
                            if 0 <= next_index < len(straight_line):
                                next_next_next_cell = straight_line[next_next_index]
                                if get_cell(next_next_next_cell[0], next_next_next_cell[1]) == 0:
                                    if i == 0:
                                        legitimate_three_ball_moves[trio].append((next_cell, trio[0],trio[1]))
                                    else:
                                        legitimate_three_ball_moves[trio].append((next_cell, trio[2],trio[1]))

                i+=1 
        # Check both directions for each trio
        for trio in adjacent_trios:
            check_move_direction(trio, -1)
            check_move_direction(trio, 1)

        def parallel_moves(trio):
                
                # the difference for parallel moves between two and three balls is that either we find 2 cells on adjacent straight line that are adjacent to the trio
                # or we can find 2 cell on adjacent straight line that is adjacent to the middle ball of the trio 



                straight_line = adjacent_trios_dict[trio]
                # find the two cells which are adjacent to both the balls in the pair
                adjacent_cells = []
                cell1, cell2, cell3 = trio

                print(f'Trio: {trio}, cell1: {cell1}, cell2: {cell2}, cell3: {cell3}')
                
                for adj_cell in adjacent_cells_dict[cell2]:
                    if adj_cell in adjacent_cells_dict[cell1] or adj_cell in adjacent_cells_dict[cell3]:
                        adjacent_cells.append(adj_cell)

                adjacent_pairs = []

                # Finding pairs among the adjacent cells
                for i in range(len(adjacent_cells)):
                    for j in range(i + 1, len(adjacent_cells)):
                        if adjacent_cells[j] in adjacent_cells_dict[adjacent_cells[i]]:
                            adjacent_pairs.append([adjacent_cells[i], adjacent_cells[j]])

                # Printing out the results
                for duo in adjacent_pairs:
                    print(f'Duo: {duo}')
                                           
                # now we need to check whether the adjacent cells are empty, and if yes, then also one cell to the right and left since a pair
                # has 4 theorically possible parallel moves
                parralel_lines = return_parallel_lines(straight_line)

                print(f'Adjacent cells: {adjacent_cells}')

                def check_parallel_line(parralel_lines, duo):
                    
                    
                        cell1 = duo[0]
                        cell2 = duo[1]
                        if get_cell(cell1[0], cell1[1]) == 0 and get_cell(cell2[0], cell2[1]) == 0:
                            # if the cells are empty, then we need to check the cells to the right and left of it but first check on which
                            # of the parallel lines it lies

                            for line in parralel_lines:
                                if cell1 in line:
                                    parallel_line = line
                                    break
                            
                            # now we need to find the index of the cell in the parallel line
                            index1 = parallel_line.index(cell1)
                            index2 = parallel_line.index(cell2)
                            # now we need to check the cells to the right and left of the cell
                            if index1 > 0:
                                if get_cell(parallel_line[index1-1][0], parallel_line[index1-1][1]) == 0:
                                    legitimate_three_ball_moves[trio].append((parallel_line[index1-1], cell1, cell2))

                            if index2 < len(parallel_line) - 1:
                                if get_cell(parallel_line[index2+1][0], parallel_line[index2+1][1]) == 0:
                                    legitimate_three_ball_moves[trio].append((cell1, cell2, parallel_line[index2+1]))

                
                for duo in adjacent_pairs:
                    print('duo that is being checked: ', duo)
                    check_parallel_line(parralel_lines, duo)

        for trio in adjacent_trios:
            parallel_moves(trio)

        return legitimate_three_ball_moves

def create_adjacent_trios_to_straight_lines(straight_lines):
    adjacent_trios_dict = {}
    
    # Function to create trios of adjacent cells
    def create_trios(line):
        for i in range(len(line) - 2):
            trio = (line[i], line[i + 1], line[i + 2])
            # Add the trio to the dictionary
            adjacent_trios_dict[trio] = line

    # Process each line
    for line in straight_lines:
        create_trios(line)
        
    return adjacent_trios_dict

print(adjacent_cells_dict)