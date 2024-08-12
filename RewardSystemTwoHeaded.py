import numpy as np
from GameRL import GameRL
from Board import Board 
import itertools
from GameOpsRL import GameOpsRL

class RewardSystemTwoHeaded:
    def __init__(self, player1, player2, push_off_reward=1000, win_reward=10000, cluster_reward=5, center_move_reward=3,
                 isolation_reward=5, self_isolation_penalty = -5, threaten_reward=7, exposure_penalty=-3, blocking_penalty=-2, multiple_threat_reward=1600, push_reward = 300, 
                 max_moves = 300, repeated_move_penalty = -3, ball_lost_penalty = -1000, ball_pushed_penalty = -50, ball_defesive_cluster_reward = 4, escaped_push_off_reward = 500, 
                 two_ball_move_reward = 3, three_ball_move_reward = 6, early_game_one_ball_move_penalty = -20, mid_game_one_ball_move_penalty = -3, late_game_one_ball_move_penalty = 0, 
                 lost_game_penalty = -1000):
        self.game = GameRL(player1, player2)
        self.game_ops = GameOpsRL(player1, player2)
        self.push_off_reward = push_off_reward
        self.win_reward = win_reward
        self.cluster_reward = cluster_reward
        self.center_move_reward = center_move_reward
        self.isolation_reward = isolation_reward
        self.self_isolation_penalty = self_isolation_penalty
        self.threaten_reward = threaten_reward
        self.exposure_penalty = exposure_penalty
        self.blocking_penalty = blocking_penalty
        self.multiple_threat_reward = multiple_threat_reward
        self.move_count = 0
        self.push_reward = push_reward
        self.max_moves = max_moves
        self.previous_move = None
        self.repeated_move_penalty = repeated_move_penalty
        self.ball_lost_penalty = ball_lost_penalty
        self.ball_pushed_penalty = ball_pushed_penalty
        self.ball_defensive_cluster_reward = ball_defesive_cluster_reward
        self.escaped_push_off_reward = escaped_push_off_reward
        self.lost_game_penalty = lost_game_penalty
        self.two_ball_move_reward = two_ball_move_reward
        self.three_ball_move_reward = three_ball_move_reward
        self.early_game_one_ball_move_penalty = early_game_one_ball_move_penalty
        self.mid_game_one_ball_move_penalty = mid_game_one_ball_move_penalty
        self.late_game_one_ball_move_penalty = late_game_one_ball_move_penalty

        self.reward_counters_offensive = {
            'push_off': 0,
            'center_move': 0,
            'cluster': 0,
            'isolation': 0,
            'self_isolation': 0,
            'threaten': 0,
            'exposure': 0,
            'blocking': 0,
            'multiple_threats': 0,
            'push': 0,
            'win': 0,
            'repeated_move': 0,
            'move_count_penalty': 0,
            'two_ball_move':0,
            'three_ball_move':0,
            'one_ball_move':0
        }

        self.reward_counters_defensive = {
            'ball_lost': 0,
            'ball_pushed': 0,
            'cluster': 0,
            'lost_game': 0,
            'escaped_push_off': 0
        }

        self.episode_reward_counters = {
            'offensive':  self.reward_counters_offensive,
            'defensive': self.reward_counters_offensive
        }

    def calculate_offensive_reward(self, current_state, next_state, balls_start, balls_end):
        reward = 0
        self.move_count += 1

        print(f"\nCalculating reward for move {self.move_count}:")

        # Reward for pushing an opponent's marble off the board
        if self.did_push_off(current_state, next_state):
            reward += self.push_off_reward
            self.reward_counters_offensive['push_off'] += 1
            print(f"Push off reward: +{self.push_off_reward}")

        # Reward for moving toward the center of the board
        if self.is_toward_center(balls_start, balls_end, self.game.current_player.color):
            reward += self.center_move_reward
            self.reward_counters_offensive['center_move'] += 1
            print(f"Center move reward: +{self.center_move_reward}")

        # Reward for clustering player's marbles when playing from behind
        if self.is_cluster_improved(current_state, next_state):
            reward += self.cluster_reward
            self.reward_counters_offensive['cluster'] += 1
            print(f"Cluster improvement reward: +{self.cluster_reward}")

        if self.is_isolated(current_state, next_state):
            reward += self.isolation_reward
            self.reward_counters_offensive['isolation'] += 1
            print(f"Isolation reward: +{self.isolation_reward}")

        if self.is_isolated_current(current_state, next_state):
            reward += self.self_isolation_penalty
            self.reward_counters_offensive['self_isolation'] += 1
            print(f"Self isolation penalty: {self.self_isolation_penalty}")

        # Reward for threatening opponent's marbles
        if self.is_threatening(current_state, next_state, balls_end):
            reward += self.threaten_reward
            self.reward_counters_offensive['threaten'] += 1
            print(f"Threaten reward: +{self.threaten_reward}")

        # Penalty for exposing own marbles
        if self.is_exposed(current_state, next_state):
            reward += self.exposure_penalty
            self.reward_counters_offensive['exposure'] += 1
            print(f"Exposure penalty: {self.exposure_penalty}")

        # Penalty for blocking own marbles
        if self.is_blocking_own_marbles(current_state, next_state):
            reward += self.blocking_penalty
            self.reward_counters_offensive['blocking'] += 1
            print(f"Blocking own marbles penalty: {self.blocking_penalty}")

        # Reward for creating multiple threatening formations
        if self.has_multiple_threats(current_state, next_state, balls_end):
            reward += self.multiple_threat_reward
            self.reward_counters_offensive['multiple_threats'] += 1
            print(f"Multiple threats reward: +{self.multiple_threat_reward}")

        push_reward = self.calculate_push_reward(current_state, next_state)
        reward += push_reward
        self.reward_counters_offensive['push'] += 1
        print(f"Push reward: +{push_reward}")

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

        # Check for winning the game
        if self.is_game_won():
            reward += self.win_reward
            self.reward_counters_offensive['win'] += 1
            print(f"Win reward: +{self.win_reward}")

        current_move = (balls_start, balls_end)
        if self.previous_move is not None:
            prev_start, prev_end = self.previous_move
            if current_move == (prev_end, prev_start):
                reward += self.repeated_move_penalty
                self.reward_counters_offensive['repeated_move'] += 1
                print(f"Repeated move penalty: {self.repeated_move_penalty}")

        self.previous_move = current_move

        move_count_penalty = 0
        if 90 <= self.move_count < 120:
            move_count_penalty = -1
        elif 120 <= self.move_count < 170:
            move_count_penalty = -3
        elif 170 <= self.move_count < 200:
            move_count_penalty = -6
        elif self.move_count >= 200:
            move_count_penalty = -15

        if move_count_penalty != 0:
            reward += move_count_penalty
            self.reward_counters_offensive['move_count_penalty'] += 1
            print(f"Move count penalty: {move_count_penalty}")

        print(f"Total offensive reward for this move: {reward}")
        for key, value in self.reward_counters_offensive.items():
            self.episode_reward_counters['offensive'][key] += value

        return reward
    
    def calculate_defensive_reward(self, current_state, next_state, balls_start, balls_end):
        
        reward = 0

        if self.game_ops.is_game_over():
            self.reward_counters_defensive['lost_game'] += 1
            return self.lost_game_penalty
        
        if self.lost_ball(current_state, next_state):
            self.reward_counters_defensive['ball_lost'] += 1
            reward+= self.ball_lost_penalty
            print(f"Ball lost reward: +{self.ball_lost_penalty}")
        
        if self.got_ball_pushed(current_state, next_state):
            self.reward_counters_defensive['ball_pushed'] += 1
            reward+= self.ball_pushed_penalty
            print(f"Ball pushed reward: +{self.ball_pushed_penalty}")
        
        if self.is_defensive_cluster_improved(current_state, next_state):  
            reward += self.ball_defensive_cluster_reward
            self.reward_counters_defensive['cluster'] += 1
            print(f"Cluster improvement reward: +{self.ball_defensive_cluster_reward}")
        
        if self.escaped_push_off_risk(current_state, next_state, balls_start, balls_end):
            reward += self.escaped_push_off_reward
            self.reward_counters_defensive['escaped_push_off'] += 1
            print(f"Escaped push off reward: +{self.escaped_push_off_reward}")
        
        print(f'Total defensive reward for this move: {reward}')

        # Update episode counters
        for key, value in self.reward_counters_defensive.items():
            self.episode_reward_counters['defensive'][key] += value

        return reward
 
    def get_and_reset_episode_counters(self):
        episode_counters = self.episode_reward_counters.copy()
        self.episode_reward_counters = {
            'offensive': {key: 0 for key in self.reward_counters_offensive},
            'defensive': {key: 0 for key in self.reward_counters_defensive}
        }
        return episode_counters

    def print_reward_summary(self):
        print("\nReward Summary for this Episode:")
        for reward_type, count in self.reward_counters.items():
            print(f"{reward_type.replace('_', ' ').title()}: {count}")

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
        
        push_penalty = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == opponent and next_grid[i][j] != opponent:
                    # Enemy ball was moved
                    push_penalty += self.push_reward
        return push_penalty
    
    
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
        
        push_reward = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == opponent and next_grid[i][j] != opponent:
                    # Enemy ball was moved
                    push_reward += self.push_reward
        return push_reward*-1
    
    def is_defensive_cluster_improved(self, current_state, next_state):
        grid, current_player = current_state
        grid_next, current_player_next = next_state

        current_player_value = current_player[0] if isinstance(current_player, list) else current_player
        
        # Get the coordinates of the player's marbles in both states
        player_marbles = [(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j] == current_player_value]    
        player_marbles_next = [(i, j) for i in range(len(grid_next)) for j in range(len(grid_next[i])) if grid_next[i][j] == current_player_value]

        def defensive_score(marbles):
            if len(marbles) < 2:
                return 0
            
            score = 0
            
            
            # Reward for marbles in the middle three rows
            middle_row = len(grid) // 2
            key_rows = [middle_row - 1, middle_row, middle_row + 1]
            center_marbles = [m for m in marbles if m[0] in key_rows]
            score += len(center_marbles) * 2
            
            # Reward for marbles forming defensive lines
            for i in range(len(marbles)):
                for j in range(i+1, len(marbles)):
                    if abs(marbles[i][0] - marbles[j][0]) <= 1 and abs(marbles[i][1] - marbles[j][1]) <= 1:
                        score += 1  # Adjacent marbles
                    if marbles[i][0] == marbles[j][0] and abs(marbles[i][1] - marbles[j][1]) == 2:
                        score += 0.5  # Marbles with one space between in the same row
            
            # Penalty for isolated marbles
            for marble in marbles:
                if not any(abs(marble[0]-m[0]) <= 1 and abs(marble[1]-m[1]) <= 1 for m in marbles if m != marble):
                    score -= 2  # Penalty for isolation
            
            # Bonus for forming a "wall" (3 or more marbles in a line)
            for row in range(len(grid)):
                row_marbles = [m for m in marbles if m[0] == row]
                if len(row_marbles) >= 3:
                    score += len(row_marbles) * 1.5
            
            return score

        current_score = defensive_score(player_marbles)
        next_score = defensive_score(player_marbles_next)

        # Defensive cluster is improved if the score increases
        return next_score > current_score
          

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

    def is_toward_center(self, balls_start, balls_end, player_color):
        
        if player_color == -1:  
            offensive_targets = [(0, 2), (1, 2), (2, 3)]  # Top side center
        else:
            offensive_targets = [(6, 3), (7, 2), (8, 2)]  # Bottom side center

        center = [4, 4]
        
        total_score = 0
        for start, end in zip(balls_start, balls_end):
            # Score for moving towards the center
            center_score = np.linalg.norm(np.array(start) - np.array(center)) - np.linalg.norm(np.array(end) - np.array(center))
            
            # Score for moving towards offensive targets
            offensive_score = 0
            for target in offensive_targets:
                offensive_score += np.linalg.norm(np.array(start) - np.array(target)) - np.linalg.norm(np.array(end) - np.array(target))
            
            # Weight offensive score higher
            total_score += center_score + 2 * offensive_score
            
            # Bonus for reaching key positions
            if end in offensive_targets:
                total_score += 5
            elif end[0] in [0, 1, 2] if player_color == 1 else [6, 7, 8]:
                total_score += 3  # Bonus for reaching opponent's side
        
        return total_score > 0

    def is_cluster_improved(self, current_state, next_state):
        
        # To assess whether the cluster improved we need to somehow assess whether after the move the balls got closer to each other
        # We can do this by calculating the distance between the balls before and after the move
        # If the distance is smaller after the move, the cluster has improved
        grid, current_player = current_state
        grid_next, current_player_next = next_state

        current_player_value = current_player[0] if isinstance(current_player, list) else current_player
        opponent = -1 if current_player_value == 1 else 1
        
        # Get the coordinates of the player's marbles in the current state
        player_marbles = [(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j] == current_player_value]    
        player_marbles_next = [(i, j) for i in range(len(grid_next)) for j in range(len(grid_next[i])) if grid_next[i][j] == current_player_value]

        def average_distance(marbles):
            if len(marbles) < 2:
                return 0
            sum_distance = 0
            count = 0
            for i in range(len(marbles)):
                for j in range(i + 1, len(marbles)):
                    sum_distance += np.linalg.norm(np.array(marbles[i]) - np.array(marbles[j]))
                    count += 1
            return sum_distance / count

        avg_distance = average_distance(player_marbles)
        avg_distance_next = average_distance(player_marbles_next)

        # Cluster improvement: less distance in the next state
        return avg_distance_next < avg_distance

    def is_game_won(self):
            
        if self.game_ops.is_game_over():
            return True
        return False
    

    def is_isolated(self, current_state, next_state):
        # Return True if any marble is isolated, False otherwise
        # Check if there is an isolated marble in the next state
        # If not, return False, but if yes, we need to see whether the isolation has increased or decreased
        grid, current_player = current_state
        grid_next, current_player_next = next_state

        current_player_value = current_player[0] if isinstance(current_player, list) else current_player
        opponent = -1 if current_player_value == 1 else 1

        # Get the coordinates of the enemy player's marbles in the current state
        player_marbles = self.game_ops.find_player_balls(grid, opponent)
        player_marbles_next = self.game_ops.find_player_balls(grid_next, opponent)

        def get_cell(board, row, col):
            if 0 <= row < len(board) and 0 <= col < len(board[row]):
                return board[row][col]
            else:
                return None

        def count_isolated_marbles(marbles, grid):
            isolated_count = 0
            for marble in marbles:
                is_isolated = True
                for marble_adjacent in self.game_ops.adjacent_cells_dict[marble]:
                    if get_cell(grid, marble_adjacent[0], marble_adjacent[1]) == opponent:
                        is_isolated = False
                        break
                if is_isolated:
                    isolated_count += 1
            return isolated_count

        # Count the number of isolated marbles in the current state
        isolated_current = count_isolated_marbles(player_marbles, grid)

        # Count the number of isolated marbles in the next state
        isolated_next = count_isolated_marbles(player_marbles_next, grid_next)

        # Return True if the number of isolated marbles has increased, False otherwise
        return isolated_next > isolated_current
    
    def is_isolated_current(self, current_state, next_state):
        # Return True if any marble is isolated, False otherwise
        # Check if there is an isolated marble in the next state
        # If not, return False, but if yes, we need to see whether the isolation has increased or decreased
        grid, current_player = current_state
        grid_next, current_player_next = next_state

        current_player_value = current_player[0] if isinstance(current_player, list) else current_player
        opponent = -1 if current_player_value == 1 else 1

        # Get the coordinates of the enemy player's marbles in the current state
        player_marbles = self.game_ops.find_player_balls(grid, current_player_value)
        player_marbles_next = self.game_ops.find_player_balls(grid_next, current_player_value)

        def get_cell(board, row, col):
            if 0 <= row < len(board) and 0 <= col < len(board[row]):
                return board[row][col]
            else:
                return None

        def count_isolated_marbles(marbles, grid):
            isolated_count = 0
            for marble in marbles:
                is_isolated = True
                for marble_adjacent in self.game_ops.adjacent_cells_dict[marble]:
                    if get_cell(grid, marble_adjacent[0], marble_adjacent[1]) == current_player_value:
                        is_isolated = False
                        break
                if is_isolated:
                    isolated_count += 1
            return isolated_count

        # Count the number of isolated marbles in the current state
        isolated_current = count_isolated_marbles(player_marbles, grid)

        # Count the number of isolated marbles in the next state
        isolated_next = count_isolated_marbles(player_marbles_next, grid_next)

        # Return True if the number of isolated marbles has increased, False otherwise
        return isolated_next > isolated_current


    def is_threatening(self, current_state, next_state, balls_end):
        # Return True if the move is threatening, False otherwise
        # Check if there are more enemy marbles in cells_on_edge in the next state than the current state

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

        # Get the coordinates of the enemy player's marbles in the current state
        enemy_marbles = self.game_ops.find_player_balls(grid, opponent)

        # Get the coordinates of the enemy player's marbles in the next state
        enemy_marbles_next = self.game_ops.find_player_balls(next_grid, opponent)

        # Count the number of enemy marbles in cells_on_edge in the current state
        current_edge_count = sum(1 for marble in enemy_marbles if marble in cells_on_edge)

        # Count the number of enemy marbles in cells_on_edge in the next state
        next_edge_count = sum(1 for marble in enemy_marbles_next if marble in cells_on_edge)

        # Return True if there are more enemy marbles in cells_on_edge in the next state than the current state
        if next_edge_count > current_edge_count:
            return True

        # First get the straight line that the balls moving are on

        # print(f'Balls end: {balls_end} for is threating')
        if len(balls_end) == 3:
            line_the_balls_moved_on = self.game.board.trios_to_straight_lines.get(tuple(balls_end))
            if line_the_balls_moved_on is None:
                line_the_balls_moved_on = self.game.board.trios_to_straight_lines.get(tuple(reversed(balls_end)))
            # print(f"Line the 3 balls {balls_end} moved on: {line_the_balls_moved_on}")
        elif len(balls_end) == 2 and isinstance(balls_end[0], tuple):
            line_the_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(balls_end))
            if line_the_balls_moved_on is None:
                line_the_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(reversed(balls_end)))

            #print(f"Line the 2 {balls_end} balls moved on: {line_the_balls_moved_on}")

        else:
            return False

        def get_cell(board, row, col):
            if 0 <= row < len(board) and 0 <= col < len(board[row]):
                return board[row][col]
            else:
                return None

        def check_if_enemy_balls_further_on(line_the_balls_moved_on, balls):
            # Find coordinates on line_the_balls_moved_on for the balls after last ball in balls_end
            last_ball = balls[0]

            #print(f"Last ball: {last_ball} in check enemy balls further on, line the balls moved on: {line_the_balls_moved_on}")
            index_of_last_ball = line_the_balls_moved_on.index(last_ball)

            # Check if there is an enemy ball further on the line and check if the number of enemy balls is less than len of balls_end so that 
            # they can be pushed off 

            count_enemy_balls = 0
            for i in range(index_of_last_ball + 1, len(line_the_balls_moved_on)):
                if get_cell(next_grid, line_the_balls_moved_on[i][0], line_the_balls_moved_on[i][1]) == opponent:
                    count_enemy_balls += 1
                    if count_enemy_balls >= len(balls_end):
                        return False

            # Check if there is an enemy ball before the first ball in balls_end
            for i in range(0, index_of_last_ball):
                if get_cell(next_grid, line_the_balls_moved_on[i][0], line_the_balls_moved_on[i][1]) == opponent:
                    count_enemy_balls += 1
                    if count_enemy_balls >= len(balls_end):
                        return False

            return count_enemy_balls > 0

        if check_if_enemy_balls_further_on(line_the_balls_moved_on, balls_end):
            return True

        # Check if the balls in balls_end connect to other friendly balls that are now threatening enemy ball to be dropped off
        def check_if_threatening_connection(balls_end, next_grid):
            for ball in balls_end:
                for adjacent in self.game_ops.adjacent_cells_dict[ball]:
                    if get_cell(next_grid, adjacent[0], adjacent[1]) == current_player_value:
                        # Now check the straight line the balls are on and whether there are enemy balls on that line that can be pushed off
                        new_connected_balls = [ball, adjacent]

                        line_the_new_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(new_connected_balls))
                        if line_the_new_balls_moved_on is None:
                            line_the_new_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(reversed(new_connected_balls)))

                        if line_the_new_balls_moved_on is None:
                            continue

                        def check_if_friendly_balls_further_on(line_the_new_balls_moved_on):
                            # Find coordinates on line_the_new_balls_moved_on for the balls after last ball in new_connected_balls
                            last_ball = new_connected_balls[-1]
                            index_of_last_ball = line_the_new_balls_moved_on.index(last_ball)

                            balls = new_connected_balls.copy()

                            if index_of_last_ball + 1 < len(line_the_new_balls_moved_on) and get_cell(next_grid, line_the_new_balls_moved_on[index_of_last_ball+1][0], line_the_new_balls_moved_on[index_of_last_ball+1][1]) == player:
                                balls.append(line_the_new_balls_moved_on[index_of_last_ball+1])
                                return balls

                            first_ball = new_connected_balls[0]
                            index_of_first_ball = line_the_new_balls_moved_on.index(first_ball)

                            if index_of_first_ball - 1 >= 0 and get_cell(next_grid, line_the_new_balls_moved_on[index_of_first_ball-1][0], line_the_new_balls_moved_on[index_of_first_ball-1][1]) == player:
                                balls.insert(0, line_the_new_balls_moved_on[index_of_first_ball-1])
                                return balls

                            return balls

                        balls = check_if_friendly_balls_further_on(line_the_new_balls_moved_on)
                        
                        print(f'the balls that we connected to are {balls}')

                        if len(balls) == 3:
                            line_key = tuple(balls)
                            line = self.game.board.trios_to_straight_lines.get(line_key)
                            if line is None:
                                line = self.game.board.trios_to_straight_lines.get(tuple(reversed(line_key)))
                            if line is not None and check_if_enemy_balls_further_on(line, balls[::-1]):
                                return True
                        elif len(balls) == 2:
                            line_key = tuple(balls)
                            line = self.game.board.pairs_to_straight_lines.get(line_key)
                            if line is None:
                                line = self.game.board.pairs_to_straight_lines.get(tuple(reversed(line_key)))
                            if line is not None and check_if_enemy_balls_further_on(line,  balls[::-1]):
                                return True

            return False

        if check_if_threatening_connection(balls_end, grid):
            return True

        return False

    def is_exposed(self, current_state, next_state):
        
        # Return True if own marbles are exposed, False otherwise
        # So it's basically the same function as is_threatening, but we need to check whether a ball on our side that was not next to the edge in the previous state

        # First hard code all the cells that are on edge

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

        # Get the coordinates of the enemy player's marbles in the current state
        enemy_marbles = self.game_ops.find_player_balls(grid, current_player_value)

        # Get the coordinates of the enemy player's marbles in the next state
        enemy_marbles_next = self.game_ops.find_player_balls(next_grid, current_player_value)

        # Count the number of enemy marbles in cells_on_edge in the current state
        current_edge_count = sum(1 for marble in enemy_marbles if marble in cells_on_edge)

        # Count the number of enemy marbles in cells_on_edge in the next state
        next_edge_count = sum(1 for marble in enemy_marbles_next if marble in cells_on_edge)

        # Return True if there are more enemy marbles in cells_on_edge in the next state than the current state
        return next_edge_count > current_edge_count

    def is_blocking_own_marbles(self, current_state, next_state):
        
        # Return True if own marbles are blocked, False otherwise
        # Check if one of the player's marbles is surrounded by the opponent's marbles in the next state

        grid, current_player  = current_state
        next_grid, next_player = next_state

        current_player_value = current_player[0] if isinstance(current_player, list) else current_player
        opponent = -1 if current_player_value == 1 else 1

        # Get the coordinates of the player's marbles in the current state

        player_marbles = self.game_ops.find_player_balls(grid, current_player_value)

        # Get the coordinates of the player's marbles in the next state

        player_marbles_next = self.game_ops.find_player_balls(next_grid, current_player_value)

        
        def get_cell(board, row, col):
            if 0 <= row < len(board) and 0 <= col < len(board[row]):
                return board[row][col]
            else:
                return None
        
        def count_blocked_marbles(marbles, grid):
            blocked_count = 0
            for marble in marbles:
                is_isolated = True
                for marble_adjacent in self.game_ops.adjacent_cells_dict[marble]:
                    if get_cell(grid, marble_adjacent[0], marble_adjacent[1]) == opponent:
                        is_isolated = False
                        break
                if is_isolated:
                    blocked_count += 1
            return blocked_count
        
        # Count the number of blocked marbles in the current state

        blocked_current = count_blocked_marbles(player_marbles, grid)

        # Count the number of blocked marbles in the next state

        blocked_next = count_blocked_marbles(player_marbles_next, next_grid)

        # Return True if the number of blocked marbles has increased, False otherwise

        return blocked_next > blocked_current
    

    def has_multiple_threats(self, current_state, next_state, balls_end):
        # Return True if the move is threatening, False otherwise
        # Check if there are more enemy marbles in cells_on_edge in the next state than the current state

        grid, player = current_state
        next_grid, next_player = next_state
        threat_count = 0

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

        # Get the coordinates of the enemy player's marbles in the current state
        enemy_marbles = self.game_ops.find_player_balls(grid, opponent)

        # Get the coordinates of the enemy player's marbles in the next state
        enemy_marbles_next = self.game_ops.find_player_balls(next_grid, opponent)

        # Count the number of enemy marbles in cells_on_edge in the current state
        current_edge_count = sum(1 for marble in enemy_marbles if marble in cells_on_edge)

        # Count the number of enemy marbles in cells_on_edge in the next state
        next_edge_count = sum(1 for marble in enemy_marbles_next if marble in cells_on_edge)

        # Return True if there are more enemy marbles in cells_on_edge in the next state than the current state
        if next_edge_count > current_edge_count + 1:
            return True
        
        elif next_edge_count > current_edge_count:
            threat_count += 1

        # First get the straight line that the balls moving are on
        if len(balls_end) == 3:
            line_the_balls_moved_on = self.game.board.trios_to_straight_lines.get(tuple(balls_end))
            if line_the_balls_moved_on is None:
                line_the_balls_moved_on = self.game.board.trios_to_straight_lines.get(tuple(reversed(balls_end)))
        elif len(balls_end) == 2:
            line_the_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(balls_end))
            if line_the_balls_moved_on is None:
                line_the_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(reversed(balls_end)))

        if line_the_balls_moved_on is None:
            return False

        def get_cell(board, row, col):
            if 0 <= row < len(board) and 0 <= col < len(board[row]):
                return board[row][col]
            else:
                return None

        def check_if_enemy_balls_further_on(line_the_balls_moved_on, balls):
            # Find coordinates on line_the_balls_moved_on for the balls after last ball in balls_end
            last_ball = balls[0]
            index_of_last_ball = line_the_balls_moved_on.index(last_ball)

            # Check if there is an enemy ball further on the line and check if the number of enemy balls is less than len of balls_end so that 
            # they can be pushed off 

            count_enemy_balls = 0
            for i in range(index_of_last_ball + 1, len(line_the_balls_moved_on)):
                if get_cell(next_grid, line_the_balls_moved_on[i][0], line_the_balls_moved_on[i][1]) == opponent:
                    count_enemy_balls += 1
                    if count_enemy_balls >= len(balls_end):
                        return False

            # Check if there is an enemy ball before the first ball in balls_end
            for i in range(0, index_of_last_ball):
                if get_cell(next_grid, line_the_balls_moved_on[i][0], line_the_balls_moved_on[i][1]) == opponent:
                    count_enemy_balls += 1
                    if count_enemy_balls >= len(balls_end):
                        return False

            return count_enemy_balls > 0

        if check_if_enemy_balls_further_on(line_the_balls_moved_on, balls_end):
            threat_count += 1

        # Check if the balls in balls_end connect to other friendly balls that are now threatening enemy ball to be dropped off
        def check_if_threatening_connection(balls_end, next_grid):
            nonlocal threat_count

            for ball in balls_end:
                for adjacent in self.game_ops.adjacent_cells_dict[ball]:
                    if get_cell(next_grid, adjacent[0], adjacent[1]) == current_player_value:
                        # Now check the straight line the balls are on and whether there are enemy balls on that line that can be pushed off
                        new_connected_balls = [ball, adjacent]

                        line_the_new_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(new_connected_balls))
                        if line_the_new_balls_moved_on is None:
                            line_the_new_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(reversed(new_connected_balls)))

                        if line_the_new_balls_moved_on is None:
                            continue

                        def check_if_friendly_balls_further_on(line_the_new_balls_moved_on):
                            # Find coordinates on line_the_new_balls_moved_on for the balls after last ball in new_connected_balls
                            last_ball = new_connected_balls[-1]
                            index_of_last_ball = line_the_new_balls_moved_on.index(last_ball)

                            balls = new_connected_balls.copy()

                            if index_of_last_ball + 1 < len(line_the_new_balls_moved_on) and get_cell(next_grid, line_the_new_balls_moved_on[index_of_last_ball+1][0], line_the_new_balls_moved_on[index_of_last_ball+1][1]) == player:
                                balls.append(line_the_new_balls_moved_on[index_of_last_ball+1])
                                return balls

                            first_ball = new_connected_balls[0]
                            index_of_first_ball = line_the_new_balls_moved_on.index(first_ball)

                            if index_of_first_ball - 1 >= 0 and get_cell(next_grid, line_the_new_balls_moved_on[index_of_first_ball-1][0], line_the_new_balls_moved_on[index_of_first_ball-1][1]) == current_player_value:
                                balls.insert(0, line_the_new_balls_moved_on[index_of_first_ball-1])
                                return balls

                            return balls

                        balls = check_if_friendly_balls_further_on(line_the_new_balls_moved_on)

                        if len(balls) == 3:
                            line_key = tuple(balls)
                            line = self.game.board.trios_to_straight_lines.get(line_key)
                            if line is None:
                                line = self.game.board.trios_to_straight_lines.get(tuple(reversed(line_key)))
                            if line is not None and check_if_enemy_balls_further_on(line, balls[::-1]):
                                threat_count += 1
                        elif len(balls) == 2:
                            line_key = tuple(balls)
                            line = self.game.board.pairs_to_straight_lines.get(line_key)
                            if line is None:
                                line = self.game.board.pairs_to_straight_lines.get(tuple(reversed(line_key, balls[::-1])))
                            if line is not None and check_if_enemy_balls_further_on(line, balls[::-1]):
                                threat_count += 1

        # Call the function to check for threatening connections
        check_if_threatening_connection(balls_end, next_grid)

        # Final return statement
        return threat_count >= 2

    