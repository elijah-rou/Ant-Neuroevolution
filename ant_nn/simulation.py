import numpy as np
import matplotlib.pyplot as plt
from ant_nn.environ.Environment import Environment
from ant_nn.agent.DeterminAnt import DeterminAnt
from ant_nn.agent.RandAnt import RandAnt
# Import error, can only be called from top level (Ant-Neuroevolution)
# if called in ant_nn, won't be able to find
class Simulation:
    def __init__(self):
        self.hi = 'hi'

    def run(self, agent_class, max_t=5000):
        '''
        INPUT:
          max_t:
        OUTPUT:
          Run the siulation for max_t steps
          return number of food retrived in each time step
        '''
        env = Environment()
        agents = [agent_class()] * 10 
        food_retrived = np.zeros(max_t)
        for t in range(max_t):
            self.env.update()
            food_retrived[t] = self.env.nest.food
        return food_retrived
    
    def sample_experiment(self):
        max_t = 3000
        agent_classes = [DeterminAnt, RandAnt]
        food_gathered = []
        t = np.arange(max_t)
        for agent_class in agent_classes:
            food_gathered.append(self.run(agent_class))
        self.plot_food(food_gathered)

    def plot_food(self, foods):  
        fig, ax = plt.subplots()
        for food in foods:
            ax.plot(food)
        fig.show()

if __name__ == 'main':
    sim = Simulation()
    sim.sample_experiment()
    print('Done')

    