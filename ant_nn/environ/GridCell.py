import numpy as np
class GridCell:
    """ Class representing a cell in the environment"""
    def __init__(
        self, row, col, kind = 'wall', food = 0
    ):
        self.active = kind == 'wall'
        self.nest = kind == 'nest'
        self.position = np.array([row, col])

        self.pheromone = 0
        self.dissapate_coef = 0.9
        
        if self.active:
            self.food = food
        else:
            self.food = 0
            
    def update(self, dt=1):
        if self.active:
            self.pheromone = self.pheromone * self.dissapate_coef ** dt
            if self.pheromone < 1e10:
                self.pheromone = 0
    
    def __str__(self):
        if self.nest: return 'N' # N for nest
        if not self.active: return 'W' # W for wall
        if self.food > 0: return 'F'
        else: return 'O'