import random
from ant_nn.agent.Agent import Agent
import numpy as np


class RandAnt(Agent):

    MAX_SPEED = 1  # maximum velocity; accessible to all agents

    def __init__(
        self,
        position=np.array((0, 0)),
    ):
        # Init
        super().__init__(None, position)
        self.food_gathered = 0
        self.distance_traveled = 0

    def update(self, env):
        """Update the Agent's state"""
        self.depositPheromone(env)
        self.move(env)

    def depositPheromone(self, env):
        """Decide whether to drop pheromone, drop it if yes"""
        i, j = self.get_coord()
        env.grid[i][j].pheromone = 1

    def move(self, env):
        """Decide a direction to move, and move"""
        # Convert speed from radian to cartetian
        abs_v = self.MAX_SPEED * np.array(
            [np.sin(self.orientation), np.cos(self.orientation)]
        )
        next_pos = self.position + abs_v
        if min(next_pos) < 0 or max(next_pos) > env.height - 1:
            self.orientation = self.orientation + np.pi
            abs_v = self.MAX_SPEED * np.array(
                (np.sin(self.orientation), np.cos(self.orientation))
            )
            next_pos = self.position + abs_v
        self.position = next_pos
        dir_change = random.random()
        if dir_change < 0.1:
            self.orientation += np.pi / 6
        elif dir_change < 0.2:
            self.orientation -= np.pi / 6
        self.pickupFood()
