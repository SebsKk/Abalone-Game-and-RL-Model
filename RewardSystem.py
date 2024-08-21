import numpy as np
from GameRL import GameRL
from Board import Board 
import itertools
from GameOpsRL import GameOpsRL

class RewardSystem:
    def __init__(self, player1, player2, push_off_reward=100, win_reward=1000, cluster_reward=5, center_move_reward=3,
                 isolation_reward=5, self_isolation_penalty = -2, threaten_reward=7, exposure_penalty=-3, blocking_penalty=-2, multiple_threat_reward=160, push_reward = 10, 
                 max_moves = 200, repeated_move_penalty = -3):
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

        self.reward_counters = {
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
            'move_count_penalty': 0
        }

    def calculate_reward(self, current_state, next_state, balls_start, balls_end):
        reward = 0
        self.move_count += 1

        print(f"\nCalculating reward for move {self.move_count}:")

        # Reward for pushing an opponent's marble off the board
        if self.did_push_off(current_state, next_state):
            reward += self.push_off_reward
            self.reward_counters['push_off'] += 1
            print(f"Push off reward: +{self.push_off_reward}")

        # Reward for moving toward the center of the board
        if self.is_toward_center(balls_start, balls_end):
            reward += self.center_move_reward
            self.reward_counters['center_move'] += 1
            print(f"Center move reward: +{self.center_move_reward}")

        # Reward for clustering player's marbles when playing from behind
        if self.is_cluster_improved(current_state, next_state):
            reward += self.cluster_reward
            self.reward_counters['cluster'] += 1
            print(f"Cluster improvement reward: +{self.cluster_reward}")

        if self.is_isolated(current_state, next_state):
            reward += self.isolation_reward
            self.reward_counters['isolation'] += 1
            print(f"Isolation reward: +{self.isolation_reward}")

        if self.is_isolated_current(current_state, next_state):
            reward += self.self_isolation_penalty
            self.reward_counters['self_isolation'] += 1
            print(f"Self isolation penalty: {self.self_isolation_penalty}")

        # Reward for threatening opponent's marbles
        if self.is_threatening(current_state, next_state, balls_end):
            reward += self.threaten_reward
            self.reward_counters['threaten'] += 1
            print(f"Threaten reward: +{self.threaten_reward}")

        # Penalty for exposing own marbles
        if self.is_exposed(current_state, next_state):
            reward += self.exposure_penalty
            self.reward_counters['exposure'] += 1
            print(f"Exposure penalty: {self.exposure_penalty}")

        # Penalty for blocking own marbles
        if self.is_blocking_own_marbles(current_state, next_state):
            reward += self.blocking_penalty
            self.reward_counters['blocking'] += 1
            print(f"Blocking own marbles penalty: {self.blocking_penalty}")

        # Reward for creating multiple threatening formations
        if self.has_multiple_threats(current_state, next_state, balls_end):
            reward += self.multiple_threat_reward
            self.reward_counters['multiple_threats'] += 1
            print(f"Multiple threats reward: +{self.multiple_threat_reward}")

        push_reward = self.calculate_push_reward(current_state, next_state)
        reward += push_reward
        self.reward_counters['push'] += 1
        print(f"Push reward: +{push_reward}")

        # Check for winning the game
        if self.is_game_won():
            reward += self.win_reward
            self.reward_counters['win'] += 1
            print(f"Win reward: +{self.win_reward}")

        current_move = (balls_start, balls_end)
        if self.previous_move is not None:
            prev_start, prev_end = self.previous_move
            if current_move == (prev_end, prev_start):
                reward += self.repeated_move_penalty
                self.reward_counters['repeated_move'] += 1
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
            self.reward_counters['move_count_penalty'] += 1
            print(f"Move count penalty: {move_count_penalty}")

        print(f"Total reward for this move: {reward}")

        return reward

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
        
        push_reward = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == opponent and next_grid[i][j] != opponent:
                    # Enemy ball was moved
                    push_reward += self.push_reward
        return push_reward

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
        
        # Check if the number of opponent's marbles has decreased
        return opponent_marbles_next < opponent_marbles_current

    def is_toward_center(self, balls_start, balls_end, center = [4,4]):
        
        # We must first set up some sort of a center to move towards
        # I think the center will be the middle cell of the middle row and the distance will be calculated from there 
        for start, end in zip(balls_start, balls_end):
            if np.linalg.norm(np.array(end) - np.array(center)) < np.linalg.norm(np.array(start) - np.array(center)):
                return True
        return False

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
        if len(balls_end) == 3:
            line_the_balls_moved_on = self.game.board.trios_to_straight_lines.get(tuple(balls_end))
            if line_the_balls_moved_on is None:
                line_the_balls_moved_on = self.game.board.trios_to_straight_lines.get(tuple(reversed(balls_end)))
            print(f"Line the 3 balls {balls_end} moved on: {line_the_balls_moved_on}")
        elif len(balls_end) == 2:
            line_the_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(balls_end))
            if line_the_balls_moved_on is None:
                line_the_balls_moved_on = self.game.board.pairs_to_straight_lines.get(tuple(reversed(balls_end)))

            print(f"Line the 2 {balls_end} balls moved on: {line_the_balls_moved_on}")

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

            print(f"Last ball: {last_ball} in check enemy balls further on, line the balls moved on: {line_the_balls_moved_on}")
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

    