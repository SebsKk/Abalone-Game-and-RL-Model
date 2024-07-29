import pygame
from GameOpsRL import GameOpsRL
import sys


class GameUI:
    def __init__(self, game_ops_rl):
        self.game = game_ops_rl
        pygame.init()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)

        # Display settings
        self.SCREEN_WIDTH = 1000
        self.SCREEN_HEIGHT = 800
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Abalone Game")
        
        self.cell_size = 50
        self.cell_gap = 5
        self.board_offset_x = (self.SCREEN_WIDTH - 9 * (self.cell_size * 2 + self.cell_gap)) // 2
        self.board_offset_y = 100

        self.font = pygame.font.Font(None, 36)

    def draw_board(self):
        self.screen.fill(self.GRAY)
        for i, row in enumerate(self.game.game.board.grid):
            offset = (9 - len(row)) * (self.cell_size + self.cell_gap) // 2
            for j, cell in enumerate(row):
                x = self.board_offset_x + offset + j * (2 * self.cell_size + self.cell_gap)
                y = self.board_offset_y + i * 1.5 * self.cell_size
                pygame.draw.circle(self.screen, self.WHITE, (x, y), self.cell_size)
                if cell == 1:
                    pygame.draw.circle(self.screen, self.RED, (x, y), self.cell_size - 4)
                elif cell == -1:
                    pygame.draw.circle(self.screen, self.BLUE, (x, y), self.cell_size - 4)

    def draw_status_bar(self):
        current_player = "Black" if self.game.game.current_player.color == 1 else "White"
        status_text = f"Current Player: {current_player} | Black Score: {self.game.game.players[0].score} | White Score: {self.game.game.players[1].score}"
        text_surface = self.font.render(status_text, True, self.BLACK)
        self.screen.blit(text_surface, (20, 20))

    def display_board(self):
        self.draw_board()
        self.draw_status_bar()
        pygame.display.flip()

    def get_cell_from_position(self, position):
        for i, row in enumerate(self.game.game.board.grid):
            offset = (9 - len(row)) * self.cell_size // 2
            for j, _ in enumerate(row):
                x = offset + j * (2 * self.cell_size + self.cell_gap)
                y = i * 1.5 * self.cell_size
                distance = ((position[0] - x)**2 + (position[1] - y)**2)**0.5
                if distance <= self.cell_size:
                    return (i, j)
        return None

    def get_human_action(self):
        self.selected_balls = []
        self.end_positions = []
        selecting_start = True
        
        confirm_button = pygame.Rect(self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 50, 130, 40)
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if confirm_button.collidepoint(pos):
                        if selecting_start and self.selected_balls:
                            selecting_start = False
                        elif not selecting_start and len(self.end_positions) == len(self.selected_balls):
                            action = {
                                'start': self.selected_balls,
                                'end': self.end_positions,
                                'type': len(self.selected_balls)
                            }
                            if self.game.step(action):
                                print(f"Action taken is valid: {action}")
                                return action
                            else:
                                print("Invalid move. Try again.")
                                self.selected_balls = []
                                self.end_positions = []
                                selecting_start = True
                    else:
                        cell = self.get_cell_from_position(pos)
                        if cell:
                            if selecting_start:
                                if len(self.selected_balls) < 3 and cell not in self.selected_balls:
                                    self.selected_balls.append(cell)
                            else:
                                if len(self.end_positions) < len(self.selected_balls):
                                    self.end_positions.append(cell)

            self.display_board()
            self.draw_selected_balls()
            self.draw_confirm_button(confirm_button, "Confirm" if selecting_start else "Move")
            pygame.display.flip()

    def draw_confirm_button(self, button, text):
        pygame.draw.rect(self.screen, self.BLACK, button)
        text_surf = self.font.render(text, True, self.WHITE)
        text_rect = text_surf.get_rect(center=button.center)
        self.screen.blit(text_surf, text_rect)

    def draw_status_bar(self):
        current_player = "Black" if self.game.game.current_player.color == 1 else "White"
        status_text = f"Current Player: {current_player} | Black Score: {self.game.game.players[0].score} | White Score: {self.game.game.players[1].score}"
        instruction_text = "Select 1-3 balls, then click 'Confirm'. Then select destination(s) and click 'Move'."
        text_surface = self.font.render(status_text, True, self.BLACK)
        instruction_surface = self.font.render(instruction_text, True, self.BLACK)
        self.screen.blit(text_surface, (20, 20))
        self.screen.blit(instruction_surface, (20, 60))

    def draw_selected_balls(self):
        for ball in self.selected_balls:
            i, j = ball
            offset = (9 - len(self.game.game.board.grid[i])) * (self.cell_size + self.cell_gap) // 2
            x = self.board_offset_x + offset + j * (2 * self.cell_size + self.cell_gap)
            y = self.board_offset_y + i * 1.5 * self.cell_size
            pygame.draw.circle(self.screen, (0, 255, 0), (x, y), self.cell_size // 2, 3)


    def show_game_over(self, winner):
        self.screen.fill(self.GRAY)
        game_over_text = self.font.render(f"Game Over! {winner} wins!", True, self.BLACK)
        restart_text = self.font.render("Press R to restart or Q to quit", True, self.BLACK)
        self.screen.blit(game_over_text, (self.SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 300))
        self.screen.blit(restart_text, (self.SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 350))
        pygame.display.flip()
        
    def run_game(self, dqn_model):

        clock = pygame.time.Clock()
        state = self.game.reset()
        done = False
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and done:
                        return True  # Restart the game
                    if event.key == pygame.K_q and done:
                        return False  # Quit the game
        
            if not done:
                    self.display_board()
                    
                    current_player = self.game.game.current_player.color

                    if current_player == 1:  # Human turn
                        action = self.get_human_action()
                        state, done = self.game.step(action)
                    else:  # AI turn
                        action_space, action_details, action_mask = self.game.get_action_space()
                        transformed_state = dqn_model.transform_state_for_nn(state)
                        action, _ = dqn_model.choose_action(transformed_state, 0, action_space, action_mask, action_details)
                        state, done = self.game.step(action)

                    print(f"Action taken: {action}")

            if done:
                winner = "Black" if self.game.get_winner().color == 1 else "White"
                self.show_game_over(winner)

                clock.tick(30)  # Limit to 30 FPS
            
        
        self.display_board()
        print("Game Over!")
        print(f"Winner: {self.game.get_winner().name}")

# Main game loop
if __name__ == "__main__":
    import torch
    from DQN import DQN
    from GameOpsRL import GameOpsRL
    from Player import Player

    player1 = Player("Black", 1)
    player2 = Player("White", -1)
    game_ops_rl = GameOpsRL(player1, player2)
    dqn_model = DQN(243, 140, game_ops_rl)

    model_path = "C:/Users/kaczm/Desktop/Abalone Project/Abalone in progress/models_deepq/20240726_172620/dqn_model_episode_3000.pth"
    dqn_model.load_state_dict(torch.load(model_path))
    dqn_model.eval()

    game_ui = GameUI(game_ops_rl)
    
    while True:
        restart = game_ui.run_game(dqn_model)
        if not restart:
            break

    pygame.quit()
    sys.exit()