import unittest
from RewardSystemTwoHeaded import RewardSystemTwoHeaded  
from Player import Player


class TestEscapedPushOffRisk(unittest.TestCase):
    def setUp(self):

        self.player1 = Player("Black", 1)
        self.player2 = Player("White", -1)
        self.game = RewardSystemTwoHeaded(self.player1, self.player2)  # Replace with your actual game class initialization
        self.initial_grid = [
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [-1,0,0,0,0,0,0,0],
            [0,-1,0,0,1,1,-1,1,-1],
            [0,1,0,0,0,0,0,0],
            [0,1,-1,-1,-1,0,0],
            [-1,1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ]

    def test_escape_from_top_edge(self):
        current_state = (self.initial_grid, -1)
        next_grid = [row[:] for row in self.initial_grid]
        next_grid[3][0] = 0
        next_grid[2][0] = -1
        print(f'next_grid: {next_grid}')
        next_state = (next_grid, 1)
        balls_start, balls_end = [(3,0)], [(2,0)]
        self.assertTrue(self.game.escaped_push_off_risk(current_state, next_state, balls_start, balls_end))

    '''def test_escape_from_bottom_edge(self):
        self.initial_grid[8][2] = 1  # Place a player's ball at the bottom edge
        current_state = (self.initial_grid, 1)
        next_grid = [row[:] for row in self.initial_grid]
        next_grid[8][2] = 0
        next_grid[7][3] = 1
        next_state = (next_grid, -1)
        balls_start, balls_end = [(8,2)], [(7,3)]
        self.assertTrue(self.game.escaped_push_off_risk(current_state, next_state, balls_start, balls_end))

    def test_escape_from_left_edge(self):
        current_state = (self.initial_grid, 1)
        next_grid = [row[:] for row in self.initial_grid]
        next_grid[3][0] = 1
        next_grid[3][1] = 0
        next_state = (next_grid, -1)
        balls_start, balls_end = [(3,0)], [(3,1)]
        self.assertTrue(self.game.escaped_push_off_risk(current_state, next_state, balls_start, balls_end))

    def test_escape_from_right_edge(self):
        self.initial_grid[3][7] = 1  # Place a player's ball at the right edge
        current_state = (self.initial_grid, 1)
        next_grid = [row[:] for row in self.initial_grid]
        next_grid[3][7] = 0
        next_grid[3][6] = 1
        next_state = (next_grid, -1)
        balls_start, balls_end = [(3,7)], [(3,6)]
        self.assertTrue(self.game.escaped_push_off_risk(current_state, next_state, balls_start, balls_end))

    def test_escape_from_corner(self):
        current_state = (self.initial_grid, 1)
        next_grid = [row[:] for row in self.initial_grid]
        next_grid[0][0] = 0
        next_grid[1][1] = 1
        next_state = (next_grid, -1)
        balls_start, balls_end = [(0,0)], [(1,1)]
        self.assertTrue(self.game.escaped_push_off_risk(current_state, next_state, balls_start, balls_end))

    def test_no_escape_needed(self):
        current_state = (self.initial_grid, 1)
        next_grid = [row[:] for row in self.initial_grid]
        next_grid[2][2] = 0
        next_grid[3][3] = 1
        next_state = (next_grid, -1)
        balls_start, balls_end = [(2,2)], [(3,3)]
        self.assertFalse(self.game.escaped_push_off_risk(current_state, next_state, balls_start, balls_end))

    def test_multiple_ball_move(self):
        current_state = (self.initial_grid, 1)
        next_grid = [row[:] for row in self.initial_grid]
        next_grid[0][0] = 0
        next_grid[0][1] = 0
        next_grid[1][0] = 1
        next_grid[1][1] = 1
        next_state = (next_grid, -1)
        balls_start, balls_end = [(0,0), (0,1)], [(1,0), (1,1)]
        self.assertTrue(self.game.escaped_push_off_risk(current_state, next_state, balls_start, balls_end))

    def test_move_into_risk(self):
        current_state = (self.initial_grid, 1)
        next_grid = [row[:] for row in self.initial_grid]
        next_grid[2][2] = 0
        next_grid[0][0] = 1
        next_state = (next_grid, -1)
        balls_start, balls_end = [(2,2)], [(0,0)]
        self.assertFalse(self.game.escaped_push_off_risk(current_state, next_state, balls_start, balls_end))

    def test_escape_from_diagonal_threat(self):
        self.initial_grid[1][0] = -1  # Place an opponent's ball diagonally adjacent
        current_state = (self.initial_grid, 1)
        next_grid = [row[:] for row in self.initial_grid]
        next_grid[0][0] = 0
        next_grid[1][1] = 1
        next_state = (next_grid, -1)
        balls_start, balls_end = [(0,0)], [(1,1)]
        self.assertTrue(self.game.escaped_push_off_risk(current_state, next_state, balls_start, balls_end))'''

if __name__ == '__main__':

    unittest.main()