from abc import ABC, abstractmethod
import numpy as np

# class Agent(ABC): <- What is ABC?
class Agent:
    """Class representing the ant agent"""

    has_food = False
    last_food_location = np.array((0, 0))
    position = np.array((0, 0))
    velocity = np.array((0, 0))

    food_gathered = 0
    distance_traveled = 0

    def __init__(self):
        self.pos = np.array((7, 0))
        self.velocity = np.array((0, 1))

    @abstractmethod
    def update(self, env):
        """ Update the Agent's state """

        # Begin Ev's code for GUI test
        next_pos = self.pos + self.velocity
        if min(next_pos) < 0 or max(next_pos) > env.height-1:
            self.velocity = self.velocity * -1
            next_pos = self.pos + self.velocity
        env.grid[self.pos[0]][self.pos[1]].pheromone = 1 # maybe move to depositePheromone
        self.pos = next_pos # maybe move to move()
        # End Ev's code for GUI test

        # raise NotImplementedError

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
