from Game import Game
from Player import Player
from Board import Board 


class GameOperation:

    def __init__(self, player1, player2):
        self.game = Game(player1, player2, Board(), None)  # No UI for this class

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
        

    def game_choice(self):

        choice = input("Would you like to play again? (yes/no): ").strip().lower()

        if choice == 'yes':
            # Re-initialize the game state
            self.game.initialize_game()
            # Start a new game
            game_operation = GameOperation(self.players[0], self.players[1])
            game_operation.run_manual_test()
        elif choice == 'no':
            print("Thanks for playing!")
        else:
            print("Invalid choice.")
            self.game_choice()  # Prompt again if the choice was invalid

    def run_manual_test(self):
        """Run manual tests by allowing the user to input moves."""
        while not self.is_game_over():
            print("\nCurrent Board State:")
            print(f'player1 score:{player1.score} player2 score: {player2.score}')
            self.game.board.display_board()
            print(f"\nIt's {self.game.current_player.name}'s turn")
            balls_start = input("Enter starting ball positions (row,col) separated by spaces (e.g. '1,2 2,3'): ").split()
            balls_end = input("Enter ending ball positions (row,col) separated by spaces (e.g. '1,3 2,4'): ").split()
            balls_start = [(int(b.split(',')[0]), int(b.split(',')[1])) for b in balls_start]
            balls_end = [(int(b.split(',')[0]), int(b.split(',')[1])) for b in balls_end]
            if len(balls_end) == len(balls_start):
                balls_start, balls_end = self.sort_balls(balls_start, balls_end)
                print(balls_start, balls_end)
                success = self.make_move(balls_start, balls_end)
                if not success:
                    print("Invalid move! Try again.")
            else: 
                print("Invalid move! Try again.")
        winner = self.get_winner()
        if winner:
            print(f"\nGame Over! {winner.name} wins!")
            self.game_choice()
        else:
            print("\nGame Over! It's a draw!")

player1 = Player("Player 1", 1)
player2 = Player("Player 2", -1)
game_operation = GameOperation(player1, player2)
game_operation.run_manual_test()