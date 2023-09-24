from GameRL import GameRL
from Player import Player
from Board import Board 
from gym import spaces

class GameOpsRL:
    def __init__(self, player1, player2):
        self.game = GameRL(player1, player2, Board())
        self.current_player = player1
        self.reward_structure = {
            "win": 100,
            "lose": -100,
            "invalid_move": -5,
            "valid_move": 1
        }
        self.observation_space = spaces.Box(low=-1, high=1, shape=(len(self.game.board.grid), len(max(self.game.board.grid, key=len))), dtype=int)


    def reset(self):
        self.game.initialize_game()
        return self.get_current_state()

    def step(self, action):
        balls_start, balls_end = action
        success = False
        if len(balls_end) == len(balls_start):
            balls_start, balls_end = self.sort_balls(balls_start, balls_end)
            success = self.make_move(balls_start, balls_end)
        else:
            # If balls_start and balls_end have different lengths, treat as an invalid move
            success = False
        
        # Define rewards
        reward = self.reward_structure["valid_move"] if success else self.reward_structure["invalid_move"]
        
        # Check if the game is over
        done = self.is_game_over()
        if done:
            winner = self.get_winner()
            if winner == self.current_player:
                reward = self.reward_structure["win"]
            else:
                reward = self.reward_structure["lose"]

        return self.get_current_state(), reward, done
    
    def get_action_space(self):

        action_space = []
        # get cell positions of all current player's balls
        
        
        # first get all legitimate one ball moves


        # then get all 2 ball moves

        # then get 3 ball moves

        
    def get_current_state(self):
        """Return the current state of the board."""
        return self.game.board.grid

    def make_move(self, balls_start, balls_end):
        """Attempt to make a move and return if the move was successful."""
        return self.game.make_move(balls_start, balls_end)

    def is_game_over(self):
        """Check if the game is over."""
        return any(player.score == 6 for player in self.game.players)

    def get_winner(self):
        """Return the player who has won the game or None if there's no winner yet."""
        for player in self.game.players:
            if player.score == 6:
                return player
        return None
    
    def sort_balls(self, balls_start, balls_end):
        if len(balls_start) > 1:
            set1 = set(balls_start)
            set2 = set(balls_end)
            common_elements = set1.intersection(set2)
            if len(common_elements) > 0:
                # Find the unique elements
                unique_start = list(set1 - set2)[0]
                unique_end = list(set2 - set1)[0]

                def distance_to_unique_end(ball):
                    return abs(ball[0] - unique_end[0]) + abs(ball[1] - unique_end[1])

                sorted_balls_start = sorted([ball for ball in balls_start if ball != unique_start], key=distance_to_unique_end)
                sorted_balls_end = sorted([ball for ball in balls_end if ball != unique_end], key=distance_to_unique_end)
            
                sorted_balls_start.append(unique_start)
                sorted_balls_end.insert(0, unique_end)
            
                return sorted_balls_start, sorted_balls_end
            else:
                if len(balls_start) == 2:
                    direction = (balls_start[0][0] - balls_start[1][0], balls_start[0][1] - balls_start[1][1])
                    if (balls_end[0][0] - balls_end[1][0], balls_end[0][1] - balls_end[1][1]) == direction:
                        return balls_start, balls_end 
                    else:
                        balls_end = balls_end[::-1]
                        return balls_start, balls_end
                else:
                    # if there are 3 balls moving parallel we need to sort them with the one having middle row/ column in the middle
                    # and then sort balls_end also with the middle one in the middle and the first one being closest to the first one in balls_start
                    sorted_balls_start = sorted(balls_start, key=lambda x: (x[0], x[1]))
                    middle_ball = sorted_balls_start[1]
                    sorted_balls_end = sorted(balls_end, key=lambda x: abs(x[0] - middle_ball[0]) + abs(x[1] - middle_ball[1]))
                    
                    # Ensure that the first ball in balls_end is closest to the first ball in balls_start
                    if abs(sorted_balls_end[0][0] - sorted_balls_start[0][0]) + abs(sorted_balls_end[0][1] - sorted_balls_start[0][1]) > \
                    abs(sorted_balls_end[-1][0] - sorted_balls_start[0][0]) + abs(sorted_balls_end[-1][1] - sorted_balls_start[0][1]):
                        sorted_balls_end = sorted_balls_end[::-1]

                    return sorted_balls_start, sorted_balls_end


        else:
            return balls_start, balls_end
        
