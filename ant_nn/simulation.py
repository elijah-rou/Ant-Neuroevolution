import numpy as np
import matplotlib.pyplot as plt
from ant_nn.environ.Environment import Environment
from ant_nn.agent.population import Population
import yaml
import time

# Import error, can only be called from top level (Ant-Neuroevolution)
# if called in ant_nn, won't be able to find
class Simulation:
    def __init__(self):
        file_stream = open("config.yaml", "r")
        config = yaml.full_load(file_stream)
        ga_config = config["population"]
        agent_params = config["agent"]["params"]

        self.epochs = config["num_epochs"]
        self.timesteps = config["num_timesteps"]
        self.runs = config["num_runs"]
        self.population = Population(
            ga_config["size"],
            ga_config["mutation_rate"],
            ga_config["mutation_strength"],
            ga_config["keep_threshold"],
            agent_params["input_size"],
            agent_params["output_size"],
            agent_params["hidden_layer_size"],
        )

    def run(self):
        """
        Run the simulation
        """
        best_scores = np.zeros(self.epochs)
        best_chromosome = []
        for e in range(self.epochs):
            t = time.strftime('%X %x %Z')
            print(f"Generation: {e+1} - {t}")
            scores = np.zeros(self.population.size())
            for i in range(self.runs):
                t = time.strftime('%X')
                print(f"Run {i+1} - {t}")
                sims = [
                    {"env": Environment(c), "food": np.zeros(self.timesteps)}
                    for c in self.population.chromosomes
                ]
                for ts in range(self.timesteps):
                    for s in sims:
                        s["env"].update()
                        s["food"][ts] = s["env"].nest.food
                scores += np.asarray([s["food"][-1] for s in sims])
            scores /= 5
            self.population.scores = scores
            self.population.makeBabies()

            best_index = np.argmax(self.population.scores)
            best_scores[e] = self.population.scores[best_index]
            print(best_scores[e])
            best_chromosome += [self.population.chromosomes[best_index]]
            print(f"Time in thread: {time.thread_time()}\n")
        return (
            best_chromosome,
            best_scores,
        )
