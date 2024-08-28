import numpy as np
from GameRL import GameRL
from Board import Board 
import itertools
from GameOpsRL import GameOpsRL

class RewardSystemTwoHeadedSimplifiedALot:


    def __init__(self, player1, player2):
        self.game = GameRL(player1, player2)
        self.game_ops = GameOpsRL(player1, player2)
        self.win_reward = 100000
        self.push_off_reward = 3000
        self.ball_lost_penalty = -2000



        self.reward_counters_offensive = {
            'push_off': 0
        }

        self.reward_counters_defensive = {
            'ball_lost': 0,

        }

        self.episode_reward_counters = {
            'offensive':  self.reward_counters_offensive,
            'defensive': self.reward_counters_offensive
        }

    def calculate_offensive_reward(self, current_state, next_state, balls_start, balls_end):
        reward = 0

        # Win reward
        if self.is_game_won():
            reward += self.win_reward

        # Push off reward
        if self.did_push_off(current_state, next_state):
            reward += self.push_off_reward
            self.reward_counters_offensive['push_off'] += 1
            print(f"Push off reward: +{self.push_off_reward}")


        return reward

    def calculate_defensive_reward(self, current_state, next_state, balls_start, balls_end):
        reward = 0

        # Ball lost penalty
        if self.lost_ball(current_state, next_state):
            reward += self.ball_lost_penalty

        return reward
    
    def get_and_reset_episode_counters(self):
        episode_counters = self.episode_reward_counters.copy()
        self.episode_reward_counters = {
            'offensive': {key: 0 for key in self.reward_counters_offensive},
            'defensive': {key: 0 for key in self.reward_counters_defensive}
        }
        return episode_counters

        

    def lost_ball(self, current_state, next_state):

        # print(f' Current state: {current_state}, next state: {next_state} in push off')
        grid, current_player = current_state

        grid_next, current_player_next = next_state
        
        current_player_value = current_player[0] if isinstance(current_player, list) else current_player
    
        # Define opponent value
        opponent = -1 if current_player_value == 1 else 1

        # Count opponent's marbles in both states
        opponent_marbles_current = sum(row.count(opponent) for row in grid)
        opponent_marbles_next = sum(row.count(opponent) for row in grid_next)
        
        # Check if the number of opponent's marbles has decreased
        return opponent_marbles_next < opponent_marbles_current
    


    def did_push_off(self, current_state, next_state):

        # print(f' Current state: {current_state}, next state: {next_state} in push off')
        grid, current_player = current_state

        grid_next, current_player_next = next_state
        
        current_player_value = current_player[0] if isinstance(current_player, list) else current_player
    
        # Define opponent value
        opponent = -1 if current_player_value == 1 else 1

        # Count opponent's marbles in both states
        opponent_marbles_current = sum(row.count(opponent) for row in grid)
        opponent_marbles_next = sum(row.count(opponent) for row in grid_next)

        # print(f'number of opponent marbles in current state: {opponent_marbles_current}, number of opponent marbles in next state: {opponent_marbles_next}')
        
        # Check if the number of opponent's marbles has decreased
        return opponent_marbles_next < opponent_marbles_current
    
    def is_game_won(self):
            
        if self.game_ops.is_game_over():
            return True
        return False
    
