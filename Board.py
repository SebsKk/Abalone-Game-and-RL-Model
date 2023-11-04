class Board:

    def __init__(self):
        self.grid = self.initialize_board()

    def initialize_board(self):

        grid = [
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

        return grid
    
    def display_board(self):
        for row in self.grid:
            print(row)

    def get_cell(self, row, col):

        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[row]):
            return self.grid[row][col]
        else:
            return None
    
    def set_cell(self, row, col, val):

        self.grid[row][col] = val

    def define_straight_lines(self):

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
        
    def check_straight_line(self, balls_start, balls_end):

        """Check if the given points lie on a straight line on the Abalone board."""
        if len(balls_start) == 1:
            return True
        else:
            all_straight_lines = self.define_straight_lines()
        
            def is_consecutive_subset(subset, mainset):
                """Check if subset is a consecutive subset of mainset."""
                if not set(subset).issubset(mainset):
                    return False
                indices = [mainset.index(point) for point in subset]
                return sorted(indices) == list(range(min(indices), min(indices) + len(subset)))

            def get_parallel_lines(line):
                """Return the parallel lines for a given line on the Abalone board."""
                line_idx = all_straight_lines.index(line)
                lines = []
                
                # If there's a line before the current line in the list, add it
                if line_idx > 0:
                    lines.append(all_straight_lines[line_idx - 1])
                
                # If there's a line after the current line in the list, add it
                if line_idx < len(all_straight_lines) - 1:
                    lines.append(all_straight_lines[line_idx + 1])
                
                return lines

            # Check for a parallel move
            if not any(ball in balls_end for ball in balls_start):
                for line in all_straight_lines:
                    if is_consecutive_subset(balls_start, line):
                        parallel_lines = get_parallel_lines(line)
                        return any(is_consecutive_subset(balls_end, pline) for pline in parallel_lines)
            else:
                start_line = next((line for line in all_straight_lines if is_consecutive_subset(balls_start, line)), None)
                return any(is_consecutive_subset(balls_start, line) for line in all_straight_lines) and \
                    any(is_consecutive_subset(balls_end, line) for line in all_straight_lines) and \
                    is_consecutive_subset(balls_end, start_line)
    
    
    def check_if_push_available(self, balls_start, balls_end):

        # if we are pushing with only one ball, return False 

        if len(balls_start) == 1:
            return False

        all_straight_lines = self.define_straight_lines()

        for line in all_straight_lines:
            if set(balls_end).issubset(set(line)):
                straight_line = line
                break

        index_of_first_moving_ball = straight_line.index(balls_end[0])
        index_of_second_moving_ball = straight_line.index(balls_end[1])

        if index_of_first_moving_ball < index_of_second_moving_ball:
            straight_line = straight_line[::-1]
            index_of_first_moving_ball, index_of_second_moving_ball = index_of_second_moving_ball, index_of_first_moving_ball

        # Now, we only need one block of code since we've handled the direction by potentially reversing the straight_line
        index_start = straight_line.index(balls_end[0])

        # Initialize counters
        player_code_start = self.get_cell(balls_start[0][0], balls_start[0][1])
        enemy_ball_count = 0

        # Count consecutive enemy balls after balls_end[0]
        for i in range(index_start + 1, len(straight_line)):
            player_code = self.get_cell(straight_line[i][0], straight_line[i][1])
            if player_code == 0:  # Empty space
                break
            if player_code == player_code_start:  
                # Same color as our balls
                print('ally ball on the line')
                return False
            if player_code != player_code_start:  # Enemy ball
                enemy_ball_count += 1

        # Check the conditions
        if enemy_ball_count >= 3 or enemy_ball_count > len(balls_start):
            print('more than 3 balls in line or not pushing with enough balls')
            return False
        return True

        
    def check_parallel(self, balls_start,balls_end):
        if len(balls_end) < 2 or balls_end[1] != balls_start[0]:
            return True
        # Check if there is any non-zero cell in the end positions
        return False
    
    def check_boundaries(self, ball):

        return 0 <= ball[0] < len(self.grid) and 0 <= ball[1] < len(self.grid[ball[0]])
    
    def is_move_valid(self, balls_start, balls_end):

        # Check if all balls are moving in a straight line
        if not self.check_straight_line(balls_start, balls_end):
            print('not straight line')
            return False
        
        print('balls straight')

        for i in range(len(balls_start)): 
            # check if we are trying to move the balls by more than 1 cell
            if abs(balls_end[i][0] - balls_start[i][0]) > 1 or abs(balls_end[i][1] - balls_start[i][1]) > 1:
                print('moving by more than 1')
                return False 
            
            # Check if the move is parallel
        if self.check_parallel(balls_start, balls_end):
            print('parallel move')
            for ball in balls_end:
                if self.get_cell(ball[0], ball[1]) != 0:
                    print('cant push on parallel move')
                    return False
        else:  # not parallel
            print('not parallel move')
            if self.get_cell(balls_end[0][0], balls_end[0][1]) != 0:
                print('trying to push ball')
                if not self.check_if_push_available(balls_start, balls_end):
                    print('push not available')
                    return False
                    
            # also check if the first ball goes out of the index
        if not self.check_boundaries(balls_end[0]): 
            print('goes out of index')
            return False
            
        print('board is_move_valid true')
        return True

