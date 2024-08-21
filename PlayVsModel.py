import pygame
from GameOpsRL import GameOpsRL

class GameUI:
    def __init__(self, game_ops_rl):
        self.game = game_ops_rl
        pygame.init()
        
        # Define colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)

        # Set up the display
        self.screen = pygame.display.set_mode((900, 800))
        pygame.display.set_caption("Abalone Game")
        self.clock = pygame.time.Clock()
        self.cell_size = 60
        self.cell_gap = 10
        self.font = pygame.font.SysFont(None, 36)

    def draw_board(self):
        self.screen.fill(self.BLACK)
        for i, row in enumerate(self.game.game.board.grid):
            offset = (9 - len(row)) * self.cell_size // 2
            for j, cell in enumerate(row):
                x = offset + j * (2 * self.cell_size + self.cell_gap)
                y = i * 1.5 * self.cell_size
                pygame.draw.circle(self.screen, self.WHITE, (x, y), self.cell_size, 1)
                if cell == 1:
                    pygame.draw.circle(self.screen, self.RED, (x, y), self.cell_size - 2)
                elif cell == -1:
                    pygame.draw.circle(self.screen, self.BLUE, (x, y), self.cell_size - 2)

    def display_board(self):
        self.draw_board()
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
        while True:
            try:
                start_input = input("Enter start position(s) (row,col for each ball, separated by semicolons): ")
                start = [tuple(map(int, pos.split(','))) for pos in start_input.split(';')]
                
                end_input = input("Enter end position(s) (row,col for each ball, separated by semicolons): ")
                end = [tuple(map(int, pos.split(','))) for pos in end_input.split(';')]
                
                if len(start) != len(end):
                    print("The number of start and end positions must be the same. Try again.")
                    continue

                # Determine the move type based on the number of balls
                if len(start) == 1:
                    move_type = 1  # Single ball move
                elif len(start) == 2:
                    move_type = 2  # Two-ball move
                elif len(start) == 3:
                    move_type = 3  # Three-ball move
                else:
                    print("Invalid number of balls selected. Choose 1, 2, or 3 balls. Try again.")
                    continue

                action = {'start': start, 'end': end, 'type': move_type}
                
                # Validate the move
                if self.game.step(action):

                    print(f"Action taken is valid: {action}")
                    return action
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input format. Use integers for row and column. Try again.")

    def run_game(self, dqn_model):
        state = self.game.reset()
        done = False
        
        while not done:
            self.display_board()
            
            current_player = self.game.game.current_player.color
            print(f"Current player: {current_player}")

            if current_player== 1:  # Human turn
                print("Your turn (Black)")

                print(f"Current player after if statement: {self.game.game.current_player.color}")
                #print available moves
                # action_space, action_details, action_mask = self.game.get_action_space()
                # print("Available moves: {}".format(action_details))

                action = self.get_human_action()
            else:  # AI turn
                
                print("AI's turn (White)")
                print(f"Before get_action_space - Current player color: {self.game.game.current_player.color}")
                action_space, action_details, action_mask = self.game.get_action_space()
                print(f"After get_action_space - Current player color: {self.game.game.current_player.color}")
                transformed_state = dqn_model.transform_state_for_nn(state)
                action, _ = dqn_model.choose_action(transformed_state, 0, action_space, action_mask, action_details)
                state, done = self.game.step(action)
            print(f"Action taken: {action}")
            
        
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

    model_path = "C:/Users/kaczm/Desktop/Abalone Project/Abalone in progress/models_deepq/20240721_170127/dqn_model_episode_400.pth"
    dqn_model.load_state_dict(torch.load(model_path))
    dqn_model.eval()

    game_ui = GameUI(game_ops_rl)
    game_ui.run_game(dqn_model)