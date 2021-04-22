import random

from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    """Class representing the ant agent"""

    MAX_SPEED = 1  # maximum velocity; accessible to all agents

    def __init__(
        self, 
        current_cell=None, 
        sensed_cells=None, 
        position=np.array((0,0)), 
        orientation=0, 
        velocity=np.array((0,0)),
    ):
        self.has_food = False
        self.last_food_location = np.array((0, 0))
        self.position = position  # position [x,y]
        self.orientation = orientation  # angle of orientation in radians

        self.current_cell = current_cell
        self.sensed_cells = sensed_cells

        self.food_gathered = 0
        self.distance_traveled = 0

    @abstractmethod
    def update(self, env):
        """ Update the Agent's state """
        pass
    
    @abstractmethod
    def depositPheromone(self, env):
        """ Decide whether to drop pheromone, drop it if yes"""
        pass

    @abstractmethod
    def move(self, env):
        """ Decide a direction to move, and move"""
        pass

    def pickupFood(self):
        """ Pickup Food if the current cell has food """
        pass

    def dropFood(self):
        """ Drop Food if the current cell is a nest cell """
        pass
    
    def get_coord(self):
        return self.position.astype(int)
