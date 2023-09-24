from Board import Board 

class GameRL:
    def __init__(self, player1, player2, board, ui):
        self.players = [player1, player2]
        self.initialize_game()  # This will set up the board and players
        self.board = board
        self.ui = ui
        # Current player is set to be player1 if player1's color is -1; otherwise, player2
        self.current_player = player1 if player1.color == -1 else player2

    def check_if_ball_pushed_off(self, balls_end):

        all_straight_lines = self.board.define_straight_lines()

        straight_line = None
        for line in all_straight_lines:
            if set(balls_end).issubset(set(line)):
                straight_line = line
                break

        if straight_line is not None:
            if straight_line.index(balls_end[0]) >= len(straight_line) - 2 or straight_line.index(balls_end[0]) <= 1:
                self.current_player.update_score(1)
                print(self.current_player.score)
                return True

        return False

    def update_pushed_ball_color(self, balls_end):

        all_straight_lines = self.board.define_straight_lines()
        for line in all_straight_lines:
            if set(balls_end).issubset(set(line)):
                straight_line = line
                first_end_ball  = straight_line.index(balls_end[0])
                second_end_ball  = straight_line.index(balls_end[1])
                if first_end_ball > second_end_ball:
                    cell_enemy_ball_pushed_to = first_end_ball + 1
                    ball_enemy_ball_pushed = straight_line[cell_enemy_ball_pushed_to]
                else:
                    cell_enemy_ball_pushed_to = first_end_ball - 1
                    ball_enemy_ball_pushed = straight_line[cell_enemy_ball_pushed_to]

        self.board.set_cell(ball_enemy_ball_pushed[0],ball_enemy_ball_pushed[1], self.current_player.color*-1) 

    def make_move(self, balls_start, balls_end):
        # If the move is valid, perform the move and switch players
        # Display the board before making the move

        if self.board.get_cell(balls_start[0][0], balls_start[0][1]) != self.current_player.color:
            return False
        
        # If the move is valid, perform the move and switch players
        if self.board.is_move_valid(balls_start, balls_end) is True:
            player_color = self.board.get_cell(balls_start[0][0],balls_start[0][1])
        
            if self.board.check_parallel(balls_start, balls_end) is True:
                # Move all balls if the balls move in a parallel manner
                for i in range(len(balls_start)):
                    self.board.set_cell(balls_end[i][0],balls_end[i][1], player_color) 
                    self.board.set_cell(balls_start[i][0],balls_start[i][1], 0)
            else:
                # If balls move in a line, push off any opponent's balls that are in the path
                if self.board.get_cell(balls_end[0][0], balls_end[0][1]) != 0:
                    if not self.check_if_ball_pushed_off(balls_end):
                        # set the cell the enemy ball was pushed to with the enemy player's color
                        self.update_pushed_ball_color(balls_end)
                # Move the balls
                for i in range(len(balls_start)):
                    self.board.set_cell(balls_end[i][0],balls_end[i][1], player_color) 
                self.board.set_cell(balls_start[len(balls_start)-1][0],balls_start[len(balls_start)-1][1], 0) 

            self.switch_player()
            return True
        else: 
            return False
    def switch_player(self):
        # Switches the current player to the other player
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]

        
    def initialize_game(self):
        # Reset the board
        # Create player instances
        
        # Reset player scores
        for player in self.players:
            player.score = 0

        self.board.initialize_board()
        