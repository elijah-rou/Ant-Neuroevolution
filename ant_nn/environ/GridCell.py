import numpy as np


class GridCell:
    """ Class representing a cell in the environment"""

    def __init__(self, row, col, kind=None, food=0, dissapate_coef=0.95):
        self.active = not kind == "wall"
        self.is_nest = kind == "nest"
        self.position = np.array([row, col])

        self.pheromone = 0
        self.dissapate_coef = dissapate_coef

        if self.active:
            self.food = food
        else:
            self.food = 0

    def update(self, dt=1):
        if self.active:
            self.pheromone = self.pheromone * self.dissapate_coef ** dt
            if self.pheromone < 1e-10:
                self.pheromone = 0

    def __str__(self):
        if not self.active:
            return "W"  # W for wall
        if self.is_nest:
            return "N"  # N for nest
        if self.food > 0:
            return "F"
        if self.pheromone > 0:
            return "P"
        else:
            return "O"
