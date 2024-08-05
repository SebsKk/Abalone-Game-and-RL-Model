import unittest
from GameRL import GameRL
from Player import Player
from Board import Board 
from collections import defaultdict
import torch
import copy
from GameOpsRL import GameOpsRL

class TestActionSpace(unittest.TestCase):
    def setUp(self):
        self.player1 = Player("Black", 1)
        self.player2 = Player("White", -1)
        self.game = GameRL(self.player1, self.player2)
        self.game_ops = GameOpsRL(self.player1, self.player2)

        self.test_grid = [
            [1,0,0,0,1],
            [1,1,1,1,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,-1,-1,-1,-1,0],
            [-1,-1,0,0,0],
        ]
        self.game.board.grid = self.test_grid
        self.game_ops.game.board.grid = self.test_grid

    def test_action_space_player1(self):
        self.game.current_player = self.player1
        self.game_ops.game.current_player = self.player1
        action_space, action_details, action_mask = self.game_ops.get_action_space()

        print("\nPlayer 1 Action Space:")
        for i, action in action_details.items():
            print(f"Action {i}: {action}")

    def test_action_space_player2(self):
        self.game.current_player = self.player2
        self.game_ops.game.current_player = self.player2
        action_space, action_details, action_mask = self.game_ops.get_action_space()

        print("\nPlayer 2 Action Space:")
        for i, action in action_details.items():
            print(f"Action {i}: {action}")

if __name__ == '__main__':
    test_case = TestActionSpace()
    test_case.setUp()
    test_case.test_action_space_player1()
    test_case.test_action_space_player2()