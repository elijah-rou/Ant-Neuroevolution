from abc import ABC, abstractmethod
import numpy as np

# from ant_nn.environ import GridCell
class Agent(ABC):
    """Class representing the ant agent"""

    def __init__(self, position=np.array((0,0)), orientation = np.array((0,0)), velocity=np.array((0,0))):
        self.has_food = False
        self.last_food_location = np.array((0, 0))
        self.position = position  # position in polar coordinates (radians)
        self.orientation = orientation  # angle of orientation in radians
        self.velocity = velocity

        self.food_gathered = 0
        self.distance_traveled = 0

    @abstractmethod
    def update(self):
        """ Update the Agent's state """
        raise NotImplementedError

    @abstractmethod
    def depositPheromone(self):
        """ Decide whether to drop pheromone """
        raise NotImplementedError

    @abstractmethod
    def move(self):
        """ Decide a direction to move """
        raise NotImplementedError

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
