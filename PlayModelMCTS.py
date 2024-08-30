import pygame
import sys
import math
import torch
from AbaloneNetAggr import AbaloneNet, MCTS
from GameOpsRL import GameOpsRL
from Player import Player
import numpy as np



class GameUI:
    def __init__(self, game_ops_rl):
        self.game = game_ops_rl
        pygame.init()
        
        # Colors
        self.BACKGROUND = (240, 240, 240)
        self.BOARD_COLOR = (139, 69, 19)
        self.CELL_COLOR = (205, 133, 63)
        self.WHITE_MARBLE = (240, 240, 240)
        self.BLACK_MARBLE = (30, 30, 30)
        self.BLACK = (0, 0, 0)
        self.HIGHLIGHT = (100, 255, 100, 128)

        # Display settings
        self.SCREEN_WIDTH = 1000
        self.SCREEN_HEIGHT = 800
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Abalone Game")
        
        # Board and marble sizes
        self.hex_size = 40
        self.marble_size = int(self.hex_size * 0.8)
        self.board_center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)

        # Font settings
        self.font = pygame.font.Font(None, 36)

        # Precompute some values for efficiency
        self.sqrt3 = math.sqrt(3)
        self.vertical_spacing = self.hex_size * 1.5

        # Initialize selected balls and end positions
        self.selected_balls = []
        self.end_positions = []

        self.board_layout = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]

        self.load_mcts_model()

    def draw_hexagon(self, surface, color, center, size):
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((center[0] + size * math.cos(angle_rad),
                           center[1] + size * math.sin(angle_rad)))
        pygame.draw.polygon(surface, color, points)

    def draw_marble(self, surface, color, center, size):
        pygame.draw.circle(surface, color, center, size)
        highlight = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(highlight, (255, 255, 255, 100), (size, size), size//2)
        surface.blit(highlight, (center[0]-size, center[1]-size))

    def draw_board(self):
        self.screen.fill(self.BACKGROUND)
        
        # Draw the main hexagonal board
        board_points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            board_points.append((
                self.board_center[0] + 5.5 * self.hex_size * math.cos(angle_rad),
                self.board_center[1] + 5.5 * self.hex_size * math.sin(angle_rad)
            ))
        pygame.draw.polygon(self.screen, self.BOARD_COLOR, board_points)

      
        
        vertical_spacing = self.hex_size * 1.5
        for row, cells in enumerate(self.board_layout):
            for col, _ in enumerate(cells):
                x = self.board_center[0] + (col - len(cells) / 2 + 0.5) * (self.hex_size * math.sqrt(3))
                y = self.board_center[1] + (row - 4) * vertical_spacing
                
                # Draw the cell
                self.draw_hexagon(self.screen, self.CELL_COLOR, (x, y), self.hex_size - 2)
                
                # Draw the marble if present
                cell_value = self.game.game.board.grid[row][col]
                if cell_value == 1:
                    self.draw_marble(self.screen, self.WHITE_MARBLE, (x, y), self.marble_size)
                elif cell_value == -1:
                    self.draw_marble(self.screen, self.BLACK_MARBLE, (x, y), self.marble_size)

        

    def get_cell_from_position(self, position):
        for row, cells in enumerate(self.board_layout):
            for col, _ in enumerate(cells):
                x = self.board_center[0] + (col - len(cells) / 2 + 0.5) * (self.hex_size * self.sqrt3)
                y = self.board_center[1] + (row - 4) * self.vertical_spacing
                
                # Calculate distance from click to cell center
                dx = position[0] - x
                dy = position[1] - y
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Check if click is within the hexagon
                if distance <= self.hex_size * 0.866:  # 0.866 is approximately sqrt(3)/2
                    return (row, col)
        return None
    
    def draw_hover_effect(self, mouse_pos):
        cell = self.get_cell_from_position(mouse_pos)
        if cell:
            row, col = cell
            x = self.board_center[0] + (col - len(self.board_layout[row]) / 2 + 0.5) * (self.hex_size * self.sqrt3)
            y = self.board_center[1] + (row - 4) * self.vertical_spacing
            
            pygame.draw.circle(self.screen, (255, 255, 255, 50), (int(x), int(y)), self.marble_size + 2)

    def draw_status_bar(self):
        current_player = "Black" if self.game.game.current_player.color == 1 else "White"
        status_text = f"Current Player: {current_player} | Black Score: {self.game.game.players[0].score} | White Score: {self.game.game.players[1].score}"
        instruction_text = "Select 1-3 balls, then click 'Confirm'. Then select destination(s) and click 'Move'."
        text_surface = self.font.render(status_text, True, self.BLACK)
        instruction_surface = self.font.render(instruction_text, True, self.BLACK)
        self.screen.blit(text_surface, (20, 20))
        self.screen.blit(instruction_surface, (20, 60))

    def display_board(self):
        self.draw_board()
        self.draw_status_bar()
        mouse_pos = pygame.mouse.get_pos()
        self.draw_hover_effect(mouse_pos)
        pygame.display.flip()

    def get_cell_from_position(self, position):
        for row, cells in enumerate(self.game.game.board.grid):
            for col, _ in enumerate(cells):
                x = self.board_center[0] + (col - len(cells) / 2 + 0.5) * (self.hex_size * 3 / 2)
                y = self.board_center[1] + (row - 4) * (self.hex_size * math.sqrt(3))
                if row % 2 == 1:
                    x += self.hex_size * 3 / 4
                distance = math.sqrt((position[0] - x)**2 + (position[1] - y)**2)
                if distance <= self.hex_size:
                    return (row, col)
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
                            balls_start, balls_end = self.game.sort_balls(action['start'], action['end'])
                            if self.game.game.board.is_move_valid(balls_start, balls_end): 
                                print(f"Action is valid: {action}")
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
                                if cell in self.selected_balls:
                                    self.selected_balls.remove(cell)
                                elif len(self.selected_balls) < 3:
                                    self.selected_balls.append(cell)
                            else:
                                if cell in self.end_positions:
                                    self.end_positions.remove(cell)
                                elif len(self.end_positions) < len(self.selected_balls):
                                    self.end_positions.append(cell)

            self.display_board()
            self.draw_selected_balls()
            self.draw_confirm_button(confirm_button, "Confirm" if selecting_start else "Move")
            pygame.display.flip()

    def draw_confirm_button(self, button, text):
        pygame.draw.rect(self.screen, self.BLACK, button)
        text_surf = self.font.render(text, True, self.WHITE_MARBLE)  # Using WHITE_MARBLE instead of WHITE
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
            self.draw_highlighted_cell(ball, (255, 255, 0, 100))  # Yellow for start positions
        for ball in self.end_positions:
            self.draw_highlighted_cell(ball, (0, 255, 0, 100))  # Green for end positions

    def draw_highlighted_cell(self, cell, color):
        row, col = cell
        x = self.board_center[0] + (col - len(self.game.game.board.grid[row]) / 2 + 0.5) * (self.hex_size * self.sqrt3)
        y = self.board_center[1] + (row - 4) * self.vertical_spacing
        
        # Draw a glowing effect
        for i in range(3):
            glow_size = self.marble_size + i * 2
            glow_color = (*color[:3], color[3] - i * 30)  # Fade out the glow
            pygame.draw.circle(self.screen, glow_color, (int(x), int(y)), glow_size)


    def show_game_over(self, winner):
        self.screen.fill(self.GRAY)
        game_over_text = self.font.render(f"Game Over! {winner} wins!", True, self.BLACK)
        restart_text = self.font.render("Press R to restart or Q to quit", True, self.BLACK)
        self.screen.blit(game_over_text, (self.SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 300))
        self.screen.blit(restart_text, (self.SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 350))
        pygame.display.flip()

    def load_mcts_model(self):
        input_dim = 243  # 9x9x3 for the board representation
        output_dim = 1686  # Adjust this if your output dimension is different
        self.mcts_model = AbaloneNet(input_dim, output_dim, self.game)
        model_path = r"C:\Users\kaczm\Desktop\Abalone Project\Abalone in progress\models_abalone_net\20240824_135154\final_abalone_mcts_model.pth"
        self.mcts_model.load_state_dict(torch.load(model_path))

        for module in self.mcts_model.modules():
            if isinstance(module, torch.nn.modules.dropout.Dropout):
                module.p = 0
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        self.mcts = MCTS(self.mcts_model.predict, self.game, c_puct=5, n_playout=300)

    def run_game(self):
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
                    state, move_valid, done, pushed_off = self.game.step(action)
                else:  # AI turn (MCTS model)
                    state = self.game.get_current_state()
                    acts, probs = self.mcts.get_move_probs(state, temp=1e-3)
                    action_space, action_details, _ = self.game.get_action_space()
                    action = self.select_action(acts, probs, action_details)
                    state, move_valid, done, pushed_off  = self.game.step(action)

                print(f"Action taken: {action}")

            if done:
                winner = "Black" if self.game.get_winner().color == 1 else "White"
                self.show_game_over(winner)

            clock.tick(30)

    def select_action(self, acts, probs, action_details):
    

        print(f'action details in select action: {action_details}')
        if not acts:
            print("No actions available. Selecting a random action from action_details.")
            # Select a random action from action_details
            available_actions = [action for action in action_details.values() if action is not None]
            if not available_actions:
                print("No available actions in action_details. Returning None.")
                return None
            return np.random.choice(available_actions)
        
        if isinstance(acts[0], int):  # If acts are indices
            valid_acts = [act for act in acts if act in action_details and action_details[act] is not None]
            actions = [action_details[act] for act in valid_acts]
            valid_probs = [p for act, p in zip(acts, probs) if act in action_details and action_details[act] is not None]
        else:
            actions = [act for act in acts if act is not None]
            valid_probs = [p for act, p in zip(acts, probs) if act is not None]
        if not actions:
            print("No actions available. Selecting a random action from action_details.")
            # Select a random action from action_details
            available_actions = [action for action in action_details.values() if action is not None]
            if not available_actions:
                print("No available actions in action_details. Returning None.")
                return None
            return np.random.choice(available_actions)
            return None

        if len(actions) != len(valid_probs):
            print(f"Mismatch between actions ({len(actions)}) and probabilities ({len(valid_probs)}). Adjusting...")
            min_len = min(len(actions), len(valid_probs))
            actions = actions[:min_len]
            valid_probs = valid_probs[:min_len]

        # Renormalize probabilities
        valid_probs = np.array(valid_probs)
        sum_probs = valid_probs.sum()
        if sum_probs == 0:
            print("All probabilities are zero. Choosing uniformly.")
            valid_probs = np.ones_like(valid_probs) / len(valid_probs)
        else:
            valid_probs /= sum_probs

        try:
            chosen_action = np.random.choice(actions, p=valid_probs)
            if chosen_action is None:
                print("Warning: Chosen action is None. Retrying selection.")
                return self.select_action(actions, valid_probs, action_details)  # Recursive call
            return chosen_action
        except ValueError as e:
            print(f"Error in np.random.choice: {e}")
            print(f"Actions: {actions}")
            print(f"Probabilities: {valid_probs}")
            return None
        

if __name__ == "__main__":

    player1 = Player("Black", -1)
    player2 = Player("White", 1)
    game_ops_rl = GameOpsRL(player1, player2)

    game_ui = GameUI(game_ops_rl)
    
    while True:
        restart = game_ui.run_game()
        if not restart:
            break

    pygame.quit()
    sys.exit()