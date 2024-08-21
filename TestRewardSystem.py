import unittest
import numpy as np
from RewardSystem import RewardSystem
from Player import Player

class TestRewardSystem(unittest.TestCase):
    def setUp(self):
        self.player1 = Player('player1', 1)
        self.player2 = Player('player2', -1)
        self.rs = RewardSystem(self.player1, self.player2)

    def test_did_push_off(self):
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
        ], -1)
        next_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,-1,0,0,0,0,0],
            [0,0,0,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], 1)
        self.assertFalse(self.rs.did_push_off(current_state, next_state))

    def test_is_toward_center(self):
        balls_start = [(0,0), (0,1), (0,2)]
        balls_end = [(1,1), (1,2), (1,3)]
        self.assertTrue(self.rs.is_toward_center(balls_start, balls_end))

    def test_is_cluster_improved(self):
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
        ], -1)
        next_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,-1,0,0,0,0,0],
            [0,0,0,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], 1)
        self.assertFalse(self.rs.is_cluster_improved(current_state, next_state))

    def test_is_isolated(self):
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
        ], -1)
        next_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,-1,0,0,0,0,0],
            [0,0,0,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], 1)
        self.assertFalse(self.rs.is_isolated(current_state, next_state))

    def test_is_threatening(self):
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
        ], -1)
        next_state = ([
            [0,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,1],
            [0,0,0,0,0,0,-1,0],
            [0,0,0,0,0,0,-1,0,0],
            [0,0,-1,0,0,-1,0,0],
            [0,0,0,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,0,0,0],
        ], 1)

        balls_end = [(5,5),(4,6),(3,6)]
        self.assertTrue(self.rs.is_threatening(current_state, next_state, balls_end))

    def test_is_exposed(self):
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
        ], -1)
        next_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,-1,0,0,0,0,0],
            [0,0,0,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], 1)
        self.assertFalse(self.rs.is_exposed(current_state, next_state))

    def test_is_blocking_own_marbles(self):
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
        ], -1)
        next_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,-1,0,0,0,0,0],
            [0,0,0,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], 1)
        self.assertFalse(self.rs.is_blocking_own_marbles(current_state, next_state))

    def test_has_multiple_threats(self):
        current_state = ([
            [1,0,0,0,1],
            [1,1,1,1,1,0],
            [0,0,1,1,1,0,0],
            [1,0,0,0,0,0,0,0],
            [0,1,1,-1,-1,-1,0,0,0],
            [0,-1,0,0,0,0,0,0],
            [0,-1,-1,-1,-1,0,0],
            [-1,-1,0,-1,-1,-1],
            [0,0,0,0,-1],
        ], -1)
        next_state = ([
            [1,0,0,0,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [1,0,0,0,0,0,0,0],
            [1,1,-1,-1,-1,0,0,0,0],
            [0,-1,0,0,0,0,0,0],
            [0,-1,-1,-1,-1,0,0],
            [-1,-1,0,-1,-1,-1],
            [0,0,0,0,-1],
        ], 1)
        balls_end = [(5,5),(4,6),(3,6)]
        self.assertTrue(self.rs.has_multiple_threats(current_state, next_state, balls_end))

    def test_calculate_reward(self):
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
        ], -1)
        next_state = ([
            [1,1,1,1,1],
            [1,1,1,1,1,1],
            [0,0,1,1,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,-1,0,0,0,0,0],
            [0,0,0,-1,-1,0,0],
            [-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1],
        ], 1)
        self.assertEqual(self.rs.calculate_reward(current_state, next_state, [(0,0), (0,1), (0,2)], [(1,1), (1,2), (1,3)]), 3)

if __name__ == '__main__':
    unittest.main()