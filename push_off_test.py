import unittest
from GameRL import GameRL  # Replace with the actual import

class TestBallPushOff(unittest.TestCase):
    def setUp(self):
        self.game = GameRL()  # Replace with your actual game class initialization
    
    def initialize_board(self):
        grid = [
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,-1],
            [0,0,-1,-1,0,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ]
        self.game.board.grid = grid

    def test_push_off_edge(self):
        self.initialize_board()
        balls_start = [(5,5), (5,6)]
        balls_end = [(5,6), (5,7)]
        self.assertTrue(self.game.check_if_ball_pushed_off(balls_start, balls_end))

    def test_no_push_off(self):
        self.initialize_board()
        balls_start = [(5,4), (5,5)]
        balls_end = [(5,5), (5,6)]
        self.assertFalse(self.game.check_if_ball_pushed_off(balls_start, balls_end))

    def test_push_against_own_color(self):
        self.initialize_board()
        balls_start = [(2,2), (2,3)]
        balls_end = [(2,3), (2,4)]
        self.assertFalse(self.game.check_if_ball_pushed_off(balls_start, balls_end))

    def test_push_off_corner(self):
        self.game.board.grid[0][4] = -1  # Place an enemy ball at the corner
        balls_start = [(0,2), (0,3)]
        balls_end = [(0,3), (0,4)]
        self.assertTrue(self.game.check_if_ball_pushed_off(balls_start, balls_end))

    def test_push_three_balls(self):
        self.initialize_board()
        self.game.board.grid[5][7] = -1  # Add another enemy ball
        balls_start = [(5,4), (5,5), (5,6)]
        balls_end = [(5,5), (5,6), (5,7)]
        self.assertTrue(self.game.check_if_ball_pushed_off(balls_start, balls_end))

    def test_push_not_enough_balls(self):
        self.initialize_board()
        self.game.board.grid[5][7] = -1  # Add another enemy ball
        balls_start = [(5,5), (5,6)]
        balls_end = [(5,6), (5,7)]
        self.assertFalse(self.game.check_if_ball_pushed_off(balls_start, balls_end))

if __name__ == '__main__':
    unittest.main()