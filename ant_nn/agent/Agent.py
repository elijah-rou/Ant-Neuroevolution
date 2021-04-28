import random

from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    """Class representing the ant agent"""

    MAX_SPEED = 1  # maximum speed; accessible to all agents

    def __init__(
        self, 
        nest_loc = [0,0],
        current_cell=None, 
        sensed_cells=[None for _ in range(5)], 
        position= [0,0], 
        orientation=0,
        has_food = False
    ):
        self.nest_loc = np.asarray(nest_loc).astype(int)
        self.has_food = has_food
        self.last_food_location = np.array((0, 0))
        self.position = np.asarray(position).astype(float)  # position [x,y]
        self.orientation = orientation  # angle of orientation in radians
        self.speed = 0

        self.current_cell = current_cell
        self.sensed_cells = sensed_cells

        self.food_gathered = 0
        self.distance_traveled = 0

    @abstractmethod
    def update(self, env):
        """ Update the Agent's state """
        raise NotImplementedError

    @abstractmethod
    def sense(self, env):
        """ Sense local environment (update current and sensed cells) """
        raise NotImplementedError
    
    @abstractmethod
    def depositPheromone(self):
        """ Decide whether to drop pheromone, drop it if yes"""
        raise NotImplementedError

    @abstractmethod
    def move(self, env):
        """ Decide a direction to move, and move"""
        raise NotImplementedError

    def pickupFood(self):
        """ Pickup Food if the current cell has food """
        if (not self.has_food) and (self.current_cell.food > 0) and (not self.current_cell.is_nest):
            self.has_food = True
            self.current_cell.food -= 1
            self.last_food_location = self.current_cell.position

    def dropFood(self):
        """ Drop Food if the current cell is a nest cell """
        if self.has_food and self.current_cell.is_nest:
            print('DROP')
            self.current_cell.food += 1
            self.has_food = False

    def get_coord(self):
        return self.position.astype(int)

    def coord_valid(self, grid, coord):
        """ Check if coordinate is on grid """
        x_valid = (coord[0] > 0) and (coord[0] < len(grid))
        y_valid = (coord[1] > 1) and (coord[1] < len(grid[0]))
        return x_valid and y_valid 
