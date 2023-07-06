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

    def check_straight_line(self, balls_start, balls_end):
        # Find the direction of the movement
        direction_to_end = (balls_end[0][0] - balls_start[0][0], balls_end[0][1] - balls_start[0][1])

        # For each ball, excluding the first one
        for i in range(1, len(balls_start)):
            # Calculate the direction from the previous ball to this ball for both start and end positions
            direction_start = (balls_start[i][0] - balls_start[i-1][0], balls_start[i][1] - balls_start[i-1][1])
            direction_end = (balls_end[i][0] - balls_end[i-1][0], balls_end[i][1] - balls_end[i-1][1])

            # If the directions don't match the overall direction, return False
            if direction_start != direction_to_end or direction_end != direction_to_end:
                return False
        # If all balls are moving in the same direction, return True
        return True
    
    def check_if_push_available(self, balls_start, balls_end):

        # if we are pushing with only one ball, return False 

        if len(balls_start) == 1:
            return False
        
        direction_start = (balls_start[0][0] - balls_start[1][0], balls_start[0][1] - balls_start[1][1])
        direction_pushing = (balls_end[0][0] - balls_start[0][0], balls_end[0][1] - balls_start[0][1])
        # now i need to check how many balls there are without a gap in the given direction, starting from balls_end[0]
        
        current_ball_against = list(balls_end[0])
        current_ball_pushing = list(balls_start[0])
        player_code_start = self.get_cell(current_ball_pushing[0], current_ball_pushing[1])
        ball_count = 1

        for i in range(len(balls_start)):


            # first check if the cell we are moving the ball to is of the same color 
            player_code_end = self.get_cell(current_ball_against[0], current_ball_against[1])
            if player_code_start == player_code_end:
                return False
            
        
            # then check if there are more balls in line 
            current_ball_against[0]  +=  direction_pushing[0]
            current_ball_against[1]  +=  direction_pushing[1]
            if self.get_cell(current_ball_against[0], current_ball_against[1]) !=  0:
                ball_count += 1
                pass 

        # if the final count of balls is 3 or more, return false
        if ball_count >= 3:
            return False 
            
        # now we need to check if the number of balls we are pushing with is larger than the number of balls being pushed
        if len(balls_start) > ball_count:
            return True
        
    def check_parallel(self, balls_start,balls_end):
        if balls_end[1] != balls_start[0]:
            return True
        # Check if there is any non-zero cell in the end positions
        for i in range(len(balls_end)):
            if self.get_cell(balls_end[i][0], balls_end[i][1]) != 0:
                return False
        return True
    
    def check_boundaries(self, ball):
        if 0 <= ball[0] < len(self.grid) and 0 <= ball[1] < len(self.grid[ball[0]]):
            return True
        return True
        
    def is_move_valid(self, balls_start, balls_end):

        # Check if all balls are moving in a straight line
        if not self.check_straight_line(balls_start, balls_end):
            return False
        
        for i in range(len(balls_start)): 
            # check if we are trying to move the balls by more than 1 cell
            if abs(balls_end[i][0] - balls_start[i][0]) > 1 or abs(balls_end[i][1] - balls_start[i][1]) > 1:
                return False 
            
            # have to check if the ball can be pushed if the cell is not empty
            # first check if it's a parallel move, and if yes, if there are any balls on the way 

            elif not self.check_parallel(balls_start,balls_end):
                return False
            
            # now check pushing moves
            elif self.get_cell(balls_end[i][0], balls_end[i][1]) != 0:
                if not self.check_if_push_available(balls_start, balls_end):
                    return False
        
            # also check if the first ball goes out of the index
            elif not self.check_boundaries(balls_end[0]): 
                return False

        return True
    

    def make_move(self, balls_start, balls_end):

        player_color = self.get_cell(balls_start[0][0],balls_start[0][1])

        if self.is_move_valid(balls_start, balls_end) is True:
            if self.check_parallel(balls_start, balls_end) is True:
                for i in range(len(balls_start)):
                    self.set_cel(balls_end[i][0],balls_end[i][1], player_color) 
                    self.set_cel(balls_start[i][0],balls_start[i][1], 0) 
            else:
                for i in range(len(balls_start)):
                    self.set_cell(balls_end[i][0],balls_end[i][1], player_color) 
                self.set_cel(balls_start[len(balls_start)-1][0],balls_start[len(balls_start)-1][1], 0) 

