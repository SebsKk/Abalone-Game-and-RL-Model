import copy

class CurriculumLearning:
    def __init__(self, difficulty_levels, episodes_per_level=200, performance_threshold=0.6):
        self.difficulty_levels = difficulty_levels
        self.current_level = 0
        self.episodes_per_level = episodes_per_level
        self.performance_threshold = performance_threshold
        self.episode_count = 0
        self.successful_episodes = 0
        self.move_count = 0 

    def get_current_difficulty(self):
        return self.difficulty_levels[self.current_level]
    
    def calculate_win_condition(self, pieces_per_player):
       
        if pieces_per_player == 14:
            return 6

        return max(2, pieces_per_player // 3)

    def adjust_board(self, original_board):
        difficulty = self.get_current_difficulty()
        pieces_per_player = difficulty['pieces_per_player']

        # Create a new board with the current difficulty settings
        new_board = copy.deepcopy(original_board)
        new_board.grid = self.adjust_pieces(new_board.grid, pieces_per_player)
        
        return new_board

    def adjust_pieces(self, grid, pieces_per_player):
        # Define standard starting positions
        standard_positions = {
            1: [(0,0), (0,1), (0,2), (0,3), (0,4),
                (1,0), (1,1), (1,2), (1,3), (1,4), (1,5),
                (2,2), (2,3), (2,4)],
            -1: [(8,0), (8,1), (8,2), (8,3), (8,4),
                 (7,0), (7,1), (7,2), (7,3), (7,4), (7,5),
                 (6,2), (6,3), (6,4)]
        }

        # Clear the board
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                grid[i][j] = 0

        # Place pieces for both players
        for player in [1, -1]:
            positions = standard_positions[player][:pieces_per_player]
            for i, j in positions:
                grid[i][j] = player

        return grid


    def update(self, total_reward, moves_made, win_condition_met):
        self.episode_count += 1

        if self.episode_count >= self.episodes_per_level:
            if self.current_level < len(self.difficulty_levels) - 1:
                self.current_level += 1
                print(f"Increasing difficulty to level {self.current_level}")
            self.episode_count = 0

    def should_increase_difficulty(self):
        return self.current_level < len(self.difficulty_levels) - 1
    

    def is_game_over(self, environment):
        current_difficulty = self.get_current_difficulty()
        win_condition = self.calculate_win_condition(current_difficulty['pieces_per_player'])
        max_moves = current_difficulty['max_moves']

        # Check if any player has reached the win condition
        if any(player.score >= win_condition for player in environment.game.players):
            return True

        # Check if the maximum number of moves has been reached
        if self.move_count >= max_moves:  # Use self.move_count instead of environment.game.move_count
            return True

        return False

    def reset_move_count(self):
        self.move_count = 0

    def increment_move_count(self):
        self.move_count += 1