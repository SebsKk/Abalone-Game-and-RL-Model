import unittest
from GameRL import GameRL  
from Player import Player
from Board  import Board

class TestBallPushOff(unittest.TestCase):
    def setUp(self):
        

        player1 = Player("Black", 1)
        player2 = Player("White", -1)
        self.board = Board()
        self.game = GameRL(player1, player2)  # Replace with your actual game class initialization
    
    def initialize_board(self):
        grid = [
            [-1,1,1,1,1],
            [-1,1,1,1,1,1],
            [1,0,1,1,1,0,0],
            [1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,-1],
            [-1,1,1, 0,0,1,1,-1],
            [0,0,-1,-1,0,0,1],
            [-1,-1,-1,-1,-1,1],
            [-1,-1,-1,-1,1],
        ]
        self.game.board.grid = grid

    def test_push_off_edge(self):
        self.initialize_board()
        balls_start = [(2,0), (3,0), (4,0)]
        balls_end = [(1,0), (2,0), (3,0)]
        self.assertTrue(self.game.check_if_ball_pushed_off(balls_start, balls_end))

   

if __name__ == '__main__':
    unittest.main()