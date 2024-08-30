from Player import Player
from Board  import Board

class GameRL:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.board = Board()
        self.initialize_game()  # This will set up the board and players
        # Current player is set to be player1 if player1's color is -1; otherwise, player2
        self.current_player = player1 if player1.color == -1 else player2

    def check_if_ball_pushed_off(self, balls_start, balls_end):

        #print('checking for ball push off')
        if len(balls_start) == 1:
            return False

        all_straight_lines = self.board.define_straight_lines()
        straight_line = None

        for line in all_straight_lines:
            if set(balls_end).issubset(set(line)):
                straight_line = line
                break

        if straight_line is None:
            return False
        
        # print(f'straight_line: {straight_line}')

        # Find the indices of the moving balls
        start_indices = [straight_line.index(ball) for ball in balls_start]
        end_indices = [straight_line.index(ball) for ball in balls_end]
        
        # Determine the direction of the move
        move_direction = 1 if min(end_indices) > min(start_indices) else -1

        player_code_start = self.board.get_cell(balls_start[0][0], balls_start[0][1])
        enemy_ball_count = 0

        # Start checking from the last ball in the direction of movement
        check_index = max(end_indices) if move_direction == 1 else min(end_indices)

        # print(f'Straight line: {straight_line}')
        # print(f'Checking from index {check_index} in direction {move_direction}')

        while 0 <= check_index < len(straight_line):
            cell = straight_line[check_index]
            player_code = self.board.get_cell(cell[0], cell[1])

            # print(f'Checking cell {cell} with player code {player_code} at index {check_index}')

            if player_code == 0:  # Empty space
                break
            if player_code == player_code_start:  
                # Same color as our balls
                if check_index not in end_indices:
                    return False
            elif player_code != player_code_start:  # Enemy ball
                enemy_ball_count += 1
                if enemy_ball_count > len(balls_start):
                    return False
                if check_index == 0 or check_index == len(straight_line) - 1:
                    # Ball is pushed off the edge
                    print(f'Ball pushed off at position {cell}')
                    self.current_player.update_score(1)
                    return True

            check_index += move_direction

        return False

    def update_pushed_ball_color(self, balls_start, balls_end):
        all_straight_lines = self.board.define_straight_lines()
        pushed_balls = []


        for line in all_straight_lines:
            if set(balls_end).issubset(set(line)):
                straight_line = line
                #print(f'Straight line: {straight_line} the balls are being pushed on')
                
                direction = 1 if line.index(balls_end[0]) > line.index(balls_start[0]) else -1
                
                # Find the index of the first ball in balls_end
                first_end_index = line.index(balls_end[0])
                
                # Check for pushed balls
                current_index = first_end_index


                #print(f'Appending ball {line[current_index + direction ]}')
                pushed_balls.append(line[current_index + direction])

                current_index += direction
                #print(f'next ball is {self.board.get_cell(line[current_index][0], line[current_index][1])}')
                if self.board.get_cell(line[current_index][0], line[current_index][1]) == self.current_player.color*-1:
                    #rint('more than one ball is being pushed')
                    #print(f'Appending ball {line[current_index + direction]}')
                    pushed_balls.append(line[current_index + direction])


        # print(f'Pushed balls: {pushed_balls}')

        for ball in pushed_balls:
            self.board.set_cell(ball[0], ball[1], self.current_player.color*-1)

        return pushed_balls

    def make_move(self, balls_start, balls_end):
        # If the move is valid, perform the move and switch players
        # Display the board before making the move

        ball_pushed_off = False

        #print('Making move')
        if isinstance(balls_start[0], int):
            balls_start = [balls_start]
            balls_end = [balls_end]

        #print(f'first moving ball: {balls_start[0]}')
        if self.board.get_cell(balls_start[0][0], balls_start[0][1]) != self.current_player.color:
            #print('Invalid move: Ball does not belong to current player')
            #print(f'Current player: {self.current_player.color}')
            #print(f'cell belongs to     : {self.board.get_cell(balls_start[0][0], balls_start[0][1])}')
            return [False, ball_pushed_off]
        
        # If the move is valid, perform the move and switch players
        if self.board.is_move_valid(balls_start, balls_end) is True:
            #print('Move is valid')
            player_color = self.board.get_cell(balls_start[0][0],balls_start[0][1])
        
            if self.board.check_parallel(balls_start, balls_end) is True:
                # Move all balls if the balls move in a parallel manner
                for i in range(len(balls_start)):
                    self.board.set_cell(balls_end[i][0],balls_end[i][1], player_color) 
                    self.board.set_cell(balls_start[i][0],balls_start[i][1], 0)
            else:
                # If balls move in a line, push off any opponent's balls that are in the path
                if self.board.get_cell(balls_end[0][0], balls_end[0][1]) != 0:
                    if not self.check_if_ball_pushed_off(balls_start, balls_end):
                        # set the cell the enemy ball was pushed to with the enemy player's color
                        # print('Updating pushed ball color')
                        self.update_pushed_ball_color(balls_start, balls_end)
                    ball_pushed_off = True
                # Move the balls
                for i in range(len(balls_start)):
                    self.board.set_cell(balls_end[i][0],balls_end[i][1], player_color) 
                self.board.set_cell(balls_start[len(balls_start)-1][0],balls_start[len(balls_start)-1][1], 0) 

            self.switch_player()

         
            # print('Player switched')
            return [True, ball_pushed_off]
        
        return False, ball_pushed_off
        '''else: 
            print(f' move is {balls_start} to {balls_end}')
            print(f'current game board is {self.board.grid}')
            print('Move is invalid')
            raise ValueError("An invalid move was detected. Stopping execution.")
            return False'''
        
    def switch_player(self):
        # Switches the current player to the other player
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]


        
    def initialize_game(self):
        # Reset the board
        # Create player instances
        
        # Reset player scores
        for player in self.players:
            player.score = 0

        self.current_player = self.players[0] if self.players[0].color == -1 else self.players[1]

        self.board.reset_board()

        
        