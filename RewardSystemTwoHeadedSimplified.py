import numpy as np
from GameRL import GameRL
from Board import Board 
import itertools
from GameOpsRL import GameOpsRL

class RewardSystemTwoHeadedSimplified:


    def __init__(self, player1, player2):
        self.game = GameRL(player1, player2)
        self.game_ops = GameOpsRL(player1, player2)
        self.win_reward = 10000
        self.push_off_reward = 3000
        self.push_enemy_reward = 700
        self.ball_pushed_penalty = -300
        self.ball_lost_penalty = -2000
        self.escaped_push_off_reward = 500
        self.one_ball_move_penalty = -10
        self.two_ball_move_reward = 15
        self.three_ball_move_reward = 45
        self.move_count = 0
        self.early_game_one_ball_move_penalty = -100
        self.mid_game_one_ball_move_penalty = -20
        self.late_game_one_ball_move_penalty = -6

        self.reward_counters_offensive = {
            'push_off': 0,
            'push': 0,
            'one_ball_move': 0,
            'two_ball_move': 0,
            'three_ball_move': 0

        }

        self.reward_counters_defensive = {
            'ball_lost': 0,
            'escaped_push_off': 0,
            'push': 0
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

        if self.calculate_push_reward(current_state, next_state):
            reward += self.push_enemy_reward
            self.reward_counters_offensive['push'] += 1
            print(f"Push reward: +{self.push_enemy_reward}")



        if isinstance(balls_start[0], tuple):  # Check if it's a multi-ball move
            if len(balls_start) == 2:
                reward += self.two_ball_move_reward
                print
                self.reward_counters_offensive['two_ball_move'] += 1
            elif len(balls_start) == 3:
                reward += self.three_ball_move_reward
                print
                self.reward_counters_offensive['three_ball_move'] += 1
        else:
            if self.move_count < 20:  # Early game
                print(f"Early game one ball move penalty: {self.early_game_one_ball_move_penalty}")
                reward += self.early_game_one_ball_move_penalty
            elif self.move_count < 100:  # Mid game
                reward += self.mid_game_one_ball_move_penalty
                print(f"Mid game one ball move penalty: {self.mid_game_one_ball_move_penalty}")
            else:  # Late game
                reward += self.late_game_one_ball_move_penalty
                print(f"Late game one ball move penalty: {self.late_game_one_ball_move_penalty}")
            self.reward_counters_offensive['one_ball_move'] += 1

        return reward

    def calculate_defensive_reward(self, current_state, next_state, balls_start, balls_end):
        reward = 0

        # Ball lost penalty
        if self.lost_ball(current_state, next_state):
            reward += self.ball_lost_penalty

        # Escaped push off reward
        if self.escaped_push_off_risk(current_state, next_state, balls_start, balls_end):
            reward += self.escaped_push_off_reward

        if self.got_ball_pushed(current_state, next_state):
            reward += self.push_enemy_reward

        return reward
    
    def get_and_reset_episode_counters(self):
        episode_counters = self.episode_reward_counters.copy()
        self.episode_reward_counters = {
            'offensive': {key: 0 for key in self.reward_counters_offensive},
            'defensive': {key: 0 for key in self.reward_counters_defensive}
        }
        return episode_counters
    
    def reset_counters(self):
        for key in self.reward_counters:
            self.reward_counters[key] = 0
        self.move_count = 0
        self.repeated_moves = {}

        
    def calculate_push_reward(self, current_state, next_state):
        grid, current_player = current_state
        next_grid, _ = next_state

        current_player_value = current_player[0] if isinstance(current_player, list) else current_player
        opponent = -1 if current_player_value == 1 else 1
        
        push_reward= 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == opponent and next_grid[i][j] != opponent:
                    # Enemy ball was moved
                    push_reward += self.push_enemy_reward
        return push_reward
    

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
    

    def got_ball_pushed(self, current_state, next_state):

        grid, current_player = current_state
        next_grid, _ = next_state

        current_player_value = current_player[0] if isinstance(current_player, list) else current_player
        opponent = -1 if current_player_value == 1 else 1
        
        push_penalty = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == opponent and next_grid[i][j] != opponent:
                    # Enemy ball was moved
                    push_penalty += self.ball_pushed_penalty
        return push_penalty*-1
    

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
    
    def escaped_push_off_risk(self, current_state, next_state, balls_start, balls_end):
        grid, player = current_state
        next_grid, next_player = next_state

        current_player_value = player[0] if isinstance(player, list) else player
        opponent = -1 if current_player_value == 1 else 1

        cells_on_edge = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (1, 5),
            (2, 0), (2, 6),
            (3, 0), (3, 7),
            (4, 0), (4, 8),
            (5, 0), (5, 7),
            (6, 0), (6, 6),
            (7, 0), (7, 5),
            (8, 0), (8, 1), (8, 2), (8, 3), (8, 4)
        ]

        def get_push_directions(ball):
            directions = []
            if ball[0] == 0:
                directions.extend([(1, 0), (1,1)])  # Down
            if ball[0] == 8:
                directions.extend([(-1, 0), (-1, 1)])  # Up
            if ball[1] == 0:
                if ball[0] < 4:
                    directions.extend([(1, 1), (0, 1)])
                elif ball[0] > 4:
                    directions.extend([(-1, 1), (0, 1)])
                else:
                    directions.extend([(-1, 0), (0, 1), (1, 0)])
            if ball[1] in [4, 5, 6, 7, 8] and (ball[0], ball[1]) in cells_on_edge:
                if ball[0] < 4:
                    directions.extend([(0, -1), (1, 0)])  # Left and Down
                elif ball[0] > 4:
                    directions.extend([(0, -1), (-1, 0)])  # Left and Up
                else:  
                    directions.extend([(-1, -1), (0, -1), (1, -1)])  # Left and Down/Up
            return directions

        def is_at_push_off_risk(ball, grid):
            if ball not in cells_on_edge:
                return False
            
            push_directions = get_push_directions(ball)
            print(f' Push directions: {push_directions} for ball {ball}')
            
            for direction in push_directions:
                print(f' Checking direction: {direction}')
                
                # Find the line that contains the ball and follows the direction
                current_line = None
                for line in self.game.board.straight_lines:
                    if ball in line:
                        # Check if the next ball in the line matches the direction
                        ball_index = line.index(ball)
                        if ball_index + 1 < len(line):
                            next_ball = line[ball_index + 1]
                            if (next_ball[0] - ball[0], next_ball[1] - ball[1]) == direction:
                                current_line = line[ball_index:]
                                break
                
                if current_line is None:
                    continue  # If no matching line found, move to next direction
                
                print(f' Current line: {current_line}')
                
                count_opponent_balls = 0
                count_own_balls = 0
                i = 0
                
                for cell in current_line[1:5]:  # Check up to 4 cells away, excluding the starting ball
                    cell_value = grid[cell[0]][cell[1]]
                    print(f' Current pos: {cell}, the ball there is {cell_value}')
                    
                    if cell_value == 0:
                        break
                    if cell_value == opponent:
                        count_opponent_balls += 1
                    elif cell_value == current_player_value and i == 0 and count_opponent_balls == 0:
                        count_own_balls += 1
                        i += 1
                    elif cell_value == current_player_value and i == 0 and count_opponent_balls != 0:
                        break
                    elif cell_value == current_player_value and i != 0:
                        i += 1
                        break
                
                if i == 0 and count_opponent_balls > 1:
                    return True
                elif i == 1 and count_opponent_balls > 2:
                    return True
            
            return False

        # Check if any ball that was at risk of being pushed off is no longer at risk after the move
        for start, end in zip(balls_start, balls_end):
            if is_at_push_off_risk(start, grid) and not is_at_push_off_risk(end, next_grid):
                return True

        # If we've reached this point, no ball has escaped the risk of being pushed off
        return False