import random

from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    """Class representing the ant agent"""
    MAX_SPEED = 1  # maximum velocity; accessible to all agents

    def __init__(
        self, 
        nest_loc = np.array((0,0)),
        current_cell=None, 
        sensed_cells=[None for i in range(5)], 
        position=np.array((0,0),dtype=float), 
        orientation=0, 
        velocity=np.array((0,0)),
    ):
        self.nest_loc = nest_loc
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
        # abs_v = self.MAX_SPEED * np.array([np.sin(self.orientation), np.cos(self.orientation)])
        # next_pos = self.position + abs_v
        # if min(next_pos) < 0 or max(next_pos) > env.height-1:
        #     self.orientation = self.orientation + np.pi
        #     abs_v = self.MAX_SPEED * np.array((np.sin(self.orientation), np.cos(self.orientation)))
        #     next_pos = self.position + abs_v
        # self.position = next_pos
        # dir_change = random.random()
        # if dir_change < 0.1:
        #     self.orientation += np.pi / 6
        # elif dir_change < 0.2:
        #     self.orientation -= np.pi / 6

    def pickupFood(self):
        """ Pickup Food if the current cell has food """
        if (not self.has_food) and (self.current_cell.food > 0) and (not self.current_cell.is_nest):
            self.has_food = True
            self.current_cell.food -= 1
            self.last_food_location = current_cell.position

    def dropFood(self):
        """ Drop Food if the current cell is a nest cell """
        if self.has_food and self.current_cell.is_nest:
            self.current_cell.food += 1
            self.has_food = False
    
    def get_coord(self):
        return self.position.astype(int)

    def coord_valid(self, grid, coord):
        """ Check if coordinate is on grid """
        x_valid = (coord[0] > 0) and (coord[0] < len(grid))
        y_valid = (coord[1] > 1) and (coord[1] < len(grid[0]))
        return x_valid and y_valid 
