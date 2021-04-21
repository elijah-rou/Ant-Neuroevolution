import Agent
import numpy as np


class DeterminAnt(Agent):
    def __init__(self):
        super.__init__()

    def update(self, current_cell, sensed_cells):
        self.current_cell = current_cell
        self.sensed_cells = sensed_cells

        self.pickupFood()
        self.dropFood()
        self.depositPheromone()
        self.move()

    def depositPheromone(self):
        if self.has_food:
            self.current_cell.pheromone += 1

    def move(self):
        if self.has_food:  # head straight to colony w/ food
            theta = np.arctan(
                self.position[1] / self.position[0]
            )  # polar angle from nest
            self.orientation = (theta + np.pi) % (2 * np.pi)
            self.speed = MAX_SPEED
        else:
            self.orientation = (self.orientation + np.random.normal(0, 0.5)) % (
                2 * np.pi
            )
            self.speed = MAX_SPEED
        self.position[0] = self.speed * cos(self.orientation)
        self.position[1] = self.speed * cos(self.orientation)
