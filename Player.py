
class Player:

    def __init__(self, name, color):
        self.name = name 
        self.color = color # 1 for player 1, -1 for player 2
        self.score = 0
    
    def update_score(self, point):

        print(f'updating score for {self.name}')
        print(f'current player score: {self.score}')
        self.score += point

