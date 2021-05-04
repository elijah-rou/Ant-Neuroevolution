import numpy as np
import matplotlib.pyplot as plt
from ant_nn.environ.Environment import Environment
from ant_nn.agent.population import Population
import yaml
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from functools import partial


file_stream = open("config.yaml", "r")
config = yaml.full_load(file_stream)
TIMESTEPS = config["num_timesteps"]

# Import error, can only be called from top level (Ant-Neuroevolution)
# if called in ant_nn, won't be able to find
def sim_env(chromosome):
    sim = {"env": Environment(chromosome), "food": np.zeros(TIMESTEPS)}
    for t in range(TIMESTEPS):
        sim["env"].update()
        sim["food"][t] = sim["env"].nest.food
    score = sim["food"][-1]
    return score

class Simulation:
    def __init__(self):
        ga_config = config["population"]
        agent_params = config["agent"]["params"]

        self.epochs = config["num_epochs"]
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

        self.executor  = ProcessPoolExecutor()
        self.scores = np.zeros(self.population.size())

    def run(self):
        """
        Run the simulation
        """
        best_scores = np.zeros(self.epochs)
        best_chromosome = []
        pop_range = range(self.population.size())
        for ep in range(self.epochs):
            t = time.strftime('%X %x %Z')
            print(f"Generation: {ep+1} - {t}")

            future_envs = {self.executor.submit(sim_env, self.population.chromosomes[i]): (i, r) for i in range(self.population.size()) for r in range(self.runs)}
            for future in as_completed(future_envs):
                chrom_index, run = future_envs[future]
                try:
                    score = future.result()
                    self.scores[chrom_index] += score
                    print(f"Chromosome {chrom_index}, run {run}: completed {score}")
                except Exception as e:
                    print(e)
            # sim_args = [c for c in self.population.chromosomes for _ in range(self.runs)]
            # for chrom_index, score in zip(pop_range, self.executor.map(sim_env, sim_args, chunksize=16)):
            #     self.scores[chrom_index%self.population.size()] += score
            #     print(f"Chromosome {chrom_index%self.population.size()}: completed {score}")
            
            self.scores = self.scores / self.runs
            self.population.scores = self.scores
            self.population.makeBabies()
            self.scores = np.zeros(self.population.size())
            best_index = np.argmax(self.population.scores)
            best_scores[ep] = self.population.scores[best_index]
            #print(best_scores[ep])
            best_chromosome += [self.population.chromosomes[best_index]]
            #print(f"Time in thread: {time.thread_time()}\n")
        return (
            best_chromosome,
            best_scores,
        )
