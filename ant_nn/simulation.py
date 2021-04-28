import numpy as np
import matplotlib.pyplot as plt
from ant_nn.environ.Environment import Environment
from ant_nn.agent.DeterminAnt import DeterminAnt
from ant_nn.agent.RandAnt import RandAnt

# Import error, can only be called from top level (Ant-Neuroevolution)
# if called in ant_nn, won't be able to find
class Simulation:
    def __init__(self, setup="default"):
        # Add Chromosome to init
        self.setup = setup

    def run(self, agent_class, max_t=1000):
        """
        INPUT:
          max_t:
        OUTPUT:
          Run the siulation for max_t steps
          return number of food retrived in each time step
        """
        env = Environment()
        if self.setup == "default":
            env.default_setup()
        else:
            env.dominant_setup()

        food_retrived = np.zeros(max_t)
        for t in range(max_t):
            env.update()
            food_retrived[t] = env.nest.food
        return food_retrived

    def sample_experiment(self):
        agent_classes = [DeterminAnt]
        food_gathered = []
        for agent_class in agent_classes:
            food_gathered.append(self.run(agent_class))
        self.plot_food(food_gathered)

    def plot_food(self, foods):
        fig, ax = plt.subplots()
        for food in foods:
            ax.plot(food)
        ax.set_title("Food v Time")
        ax.set_xlabel("time")
        ax.set_ylabel("Food Collected")
        plt.show()
