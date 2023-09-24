
class Player:

    def __init__(self, name, color):
        self.name = name 
        self.color = color # 1 for player 1, -1 for player 2
        self.score = 0
    
    def update_score(self, point):
        self.score += point

