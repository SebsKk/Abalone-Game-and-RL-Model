import pygame
from Game import Game
from Player import Player
from Board import Board 

class GameUI:

    def __init__(self, game):
        self.game = game

        pygame.init()
        
        # Define some colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        # Create a screen and a clock
        self.screen = pygame.display.set_mode((900, 800))
        pygame.display.set_caption("Abalone Game")
        self.clock = pygame.time.Clock()
        self.cell_size = 60  # Radius of each cell
        self.cell_gap = 10   # Gap between each cell
        self.selected_balls = []
        self.font = pygame.font.SysFont(None, 36)

        self.balls_end_positions = []

    def draw_board(self):
        for i, row in enumerate(self.game.board.grid):
            offset = (9 - len(row)) * self.cell_size // 2
            for j, cell in enumerate(row):
                x = offset + j * (2 * self.cell_size + self.cell_gap)
                y = i * 1.5 * self.cell_size
                pygame.draw.circle(self.screen, self.WHITE, (x, y), self.cell_size, 1)

    def draw_balls(self):
        for i, row in enumerate(self.game.board.grid):
            offset = (9 - len(row)) * self.cell_size // 2
            for j, cell in enumerate(row):
                x = offset + j * (2 * self.cell_size + self.cell_gap)
                y = i * 1.5 * self.cell_size
                if cell == 1:
                    pygame.draw.circle(self.screen, self.RED, (x, y), self.cell_size)
                elif cell == -1:
                    pygame.draw.circle(self.screen, self.BLUE, (x, y), self.cell_size)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.process_click(event.pos)

            # Here, I will add more event handling, like detecting mouse clicks to move balls.

    def ball_at_position(self, position):
        return self.game.board.get_cell(position[0], position[1])

    def process_click(self, position):
        clicked_cell = self.get_cell_from_position(position)
        clicked_ball_value = self.ball_at_position(clicked_cell)
        current_player_value = self.game.current_player.color
        print("Clicked Position:", position)
        
        # If a ball is clicked
        if clicked_cell and self.ball_at_position(clicked_cell):
            print("Clicked on a ball at position:", clicked_cell)
            
            # Toggle selection status of the clicked ball
            if clicked_cell in self.selected_balls:
                self.selected_balls.remove(clicked_cell)
            else:
                # Ensure only up to 3 balls can be selected
                if len(self.selected_balls) < 3:
                    self.selected_balls.append(clicked_cell)
                if len(self.selected_balls) == 3:
                    self.sort_selected_balls()
        
        # If a cell without a ball is clicked (i.e., potential end position)
        elif self.selected_balls and (clicked_ball_value != current_player_value):
            balls_end_position = self.get_cell_from_position(position)
            print("End position determined:", balls_end_position)
            
            if balls_end_position:
                
                self.balls_end_positions.append(balls_end_position)
                
                if len(self.balls_end_positions) == 1:
                    print('working')
                    # Calculate the direction between the end position and the first selected ball
                    direction_to_first = (self.balls_end_positions[0][0] - self.selected_balls[0][0], 
                                        self.balls_end_positions[0][1] - self.selected_balls[0][1])
                    
                    # Calculate the direction between the end position and the last selected ball
                    direction_to_last = (self.balls_end_positions[0][0] - self.selected_balls[-1][0], 
                                        self.balls_end_positions[0][1] - self.selected_balls[-1][1])
                    
                    if len(self.selected_balls) == 3:

                        print('3 chosen balls')
                    # Calculate the direction between the first and second selected balls
                        direction_first_second = (self.selected_balls[0][0] - self.selected_balls[1][0], 
                                            self.selected_balls[0][1] - self.selected_balls[1][1])
                        
                        # Calculate the direction between the last and second selected balls
                        direction_last_second= (self.selected_balls[2][0] - self.selected_balls[1][0], 
                                            self.selected_balls[2][1] - self.selected_balls[1][1])
                    
                        # Check if the move is inline
                        is_inline = direction_to_first == direction_first_second or direction_to_last == direction_last_second
                        
                        # If the direction to the last ball matches but not to the first, swap them
                        if direction_to_last == direction_last_second:
                            self.selected_balls[0], self.selected_balls[-1] = self.selected_balls[-1], self.selected_balls[0]
                        
                        # If the move is inline
                        if is_inline:
                            print(self.selected_balls)
                            print('move is inline')
                            self.send_move_to_game(self.selected_balls, self.calculate_balls_end(balls_end_position))
                            self.selected_balls = []
                            self.balls_end_positions = []
                    
                    elif len(self.selected_balls) == 2:
                        print('2 chosen balls')
                        direction_first_second = (self.selected_balls[0][0] - self.selected_balls[1][0], 
                                            self.selected_balls[0][1] - self.selected_balls[1][1])
                        
                        # Calculate the direction between the second and first selected balls
                        direction_second_first= (self.selected_balls[1][0] - self.selected_balls[0][0], 
                                            self.selected_balls[1][1] - self.selected_balls[0][1])
                    
                        # Check if the move is inline
                        is_inline = direction_to_first == direction_first_second or direction_to_last == direction_second_first
                        
                        # If the direction to the last ball matches but not to the first, swap them
                        if direction_to_last == direction_second_first:
                            self.selected_balls[0], self.selected_balls[-1] = self.selected_balls[-1], self.selected_balls[0]
                        
                        # If the move is inline
                        if is_inline:
                            print(self.selected_balls)
                            print('move is inline')
                
                            self.send_move_to_game(self.selected_balls, self.calculate_balls_end(balls_end_position))
                            self.selected_balls = []
                            self.balls_end_positions = []

                    elif len(self.selected_balls) == 1:
                        self.send_move_to_game(self.selected_balls, self.calculate_balls_end(balls_end_position))
                        self.selected_balls = []
                        self.balls_end_positions = []
                # If the move is parallel
                else:
                    # Ensure the order of end positions matches the order of selected balls
                    if len(self.selected_balls) == len(self.balls_end_positions):
                        direction_selected = (self.selected_balls[-1][0] - self.selected_balls[0][0], 
                                            self.selected_balls[-1][1] - self.selected_balls[0][1])
                        
                        # Define a function to determine how "aligned" a point is with a given direction
                        def alignment(pos, direction):
                            return pos[0]*direction[0] + pos[1]*direction[1]
                        
                        # Sort the balls_end_positions based on their alignment with the direction of selected balls
                        self.balls_end_positions.sort(key=lambda pos: alignment(pos, direction_selected))
                        
                        self.send_move_to_game(self.selected_balls, self.balls_end_positions)
                        self.selected_balls = []
                        self.balls_end_positions = []


    def sort_selected_balls(self):
        sorted_balls = [self.selected_balls[0]]
        remaining_balls = self.selected_balls[1:]

        while remaining_balls:
            last_ball = sorted_balls[-1]
            next_ball = min(remaining_balls, key=lambda ball: (ball[0]-last_ball[0])**2 + (ball[1]-last_ball[1])**2)
            sorted_balls.append(next_ball)
            remaining_balls.remove(next_ball)

        self.selected_balls = sorted_balls  
        print(sorted_balls)

    def get_cell_from_position(self, position):
        for i, row in enumerate(self.game.board.grid):
            offset = (9 - len(row)) * self.cell_size // 2
            for j, cell in enumerate(row):
                x = offset + j * (2 * self.cell_size + self.cell_gap)
                y = i * 1.5 * self.cell_size
                distance = ((position[0] - x)**2 + (position[1] - y)**2)**0.5
                if distance <= self.cell_size:
                    return (i, j)
        return None

    def calculate_balls_end(self, end_position):
        # Calculate balls_end based on the direction of the move
        direction = (end_position[0] - self.selected_balls[0][0], end_position[1] - self.selected_balls[0][1])
        if abs(direction[0]) > 1 or abs(direction[1]) > 1:
            return None  # Invalid move direction
        print([(ball[0] + direction[0], ball[1] + direction[1]) for ball in self.selected_balls])
        return [(ball[0] + direction[0], ball[1] + direction[1]) for ball in self.selected_balls]

    def send_move_to_game(self, balls_start, balls_end):
        valid_move = self.game.make_move(balls_start, balls_end)
        if valid_move:
            # If the move was valid, update the board display or do any other necessary actions
            pass


    def draw_selected_balls(self):
        """Draws the 'X' markers on selected balls"""
        for ball in self.selected_balls:
            i, j = ball
            offset = (9 - len(self.game.board.grid[i])) * self.cell_size // 2
            x = offset + j * (2 * self.cell_size + self.cell_gap)
            y = i * 1.5 * self.cell_size
            text = self.font.render('X', True, (0, 255, 0))
            self.screen.blit(text, (x - 10, y - 10))

    def game_choice(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False  # Exit the game
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    # Check if the click is within the "YES" button
                    if 300 <= x <= 500 and 300 <= y <= 350:
                        return True  # Play again
                    # Check if the click is within the "NO" button
                    if 300 <= x <= 500 and 375 <= y <= 425:
                        return False  # Exit the game

            self.draw_game_choice_buttons()
            pygame.display.flip()
    
    def draw_game_choice_buttons(self):
        self.screen.fill((0, 0, 0))  # Fill the screen with black
        
        # Draw the "YES" button
        pygame.draw.rect(self.screen, (0, 255, 0), (300, 300, 200, 50))
        yes_text = self.font.render('YES', True, (255, 255, 255))
        self.screen.blit(yes_text, (375, 310))
        
        # Draw the "NO" button
        pygame.draw.rect(self.screen, (255, 0, 0), (300, 375, 200, 50))
        no_text = self.font.render('NO', True, (255, 255, 255))
        self.screen.blit(no_text, (375, 385))



    def run(self):
        while True:
            self.screen.fill((0, 0, 0))  # Fill the screen with black to clear previous frame
            self.handle_events()
            self.draw_board()
            self.draw_balls()
            self.draw_selected_balls()  # Separate call to draw 'X' markers
            pygame.display.flip()
            self.clock.tick(60)

# When starting the game:
player1 = Player("Alice", 1)
player2 = Player("Bob", -1)
ui = GameUI(None)  # Temporarily initialize GameUI without a game instance
board = Board()  # Create an instance of the Board class
game = Game(player1, player2, board, ui)  # Provide the board instance here
ui.game = game  # Now set the game instance for the UI
game.ui = ui  # Pass the GameUI instance to the game
ui.run()