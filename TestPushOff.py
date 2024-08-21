import unittest
import numpy as np
from RewardSystem import RewardSystem
from Player import Player

class TestRewardSystem(unittest.TestCase):
    def setUp(self):
        player1 = Player("Black", 1)
        player2 = Player("White", -1)
        self.reward_system = RewardSystem(player1, player2)

    def test_did_push_off_no_change(self):
        current_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,-1,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], 1)
        next_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,-1,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], -1)
        self.assertFalse(self.reward_system.did_push_off(current_state, next_state))

    def test_did_push_off_opponent_marble_removed(self):
        current_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,-1,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], 1)
        next_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,-1,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,0],  # One opponent marble removed
        ], -1)
        self.assertTrue(self.reward_system.did_push_off(current_state, next_state))
if __name__ == '__main__':
    unittest.main()