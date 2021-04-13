import random

from abc import ABC, abstractmethod
import numpy as np

# class Agent(ABC): <- What is ABC?
class Agent:
    """Class representing the ant agent"""

    has_food = False
    last_food_location = np.array((0, 0))
    position = np.array((0, 0))
    velocity = np.array((0, 0))
    random_v = np.array((0.1, 0.1))

    food_gathered = 0
    distance_traveled = 0

    def __init__(self):
        self.pos = np.array((7, 0))
        self.dir = 0 
        self.v = 1

    @abstractmethod
    def update(self, env):
        """ Update the Agent's state """
        self.depositPheromone(env)
        self.move(env)
    
    @abstractmethod
    def depositPheromone(self, env):
        """ Decide whether to drop pheromone, drop it if yes"""
        i, j = self.get_coord()
        env.grid[i][j].pheromone = 1

    @abstractmethod
    def move(self, env):
        """ Decide a direction to move, and move"""
        abs_v = self.v * np.array([np.sin(self.dir), np.cos(self.dir)])
        next_pos = self.pos + abs_v
        if min(next_pos) < 0 or max(next_pos) > env.height-1:
            self.dir = self.dir + np.pi
            abs_v = self.v * np.array((np.sin(self.dir), np.cos(self.dir)))
            next_pos = self.pos + abs_v
        self.pos = next_pos
        dir_change = random.random()
        if dir_change < 0.1:
            self.dir += np.pi / 6
        elif dir_change < 0.2:
            self.dir -= np.pi / 6

    def pickupFood(self):
        """ Pickup Food if the current cell has food """
        if self.current_cell.food > 0:
            self.has_food = True
            cell.food -= 1
            self.last_food_location = current_cell.position

    def dropFood(self):
        """ Drop Food if the current cell is a nest cell """
        if self.has_food and self.current_cell.is_nest:
            food_gathered += 1
            self.has_food = False
    
    def get_coord(self):
        return self.pos.astype(int)
