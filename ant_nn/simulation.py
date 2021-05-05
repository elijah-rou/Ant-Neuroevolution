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


def sim_env(chromosome):
    sim = {"env": Environment(chromosome), "food": np.zeros(TIMESTEPS)}
    for t in range(TIMESTEPS):
        sim["env"].update()
        sim["food"][t] = sim["env"].nest.food
    score = sim["food"][-1]
    return score

def plot_food(foods):
        fig, ax = plt.subplots()
        for food in foods:
            ax.plot(food)
        ax.set_title("Food v Time")
        ax.set_xlabel("time")
        ax.set_ylabel("Food Collected")
        plt.show()


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

        self.executor = ProcessPoolExecutor()
        self.scores = np.zeros((self.population.size(), self.runs))

    def run(self):
        """
        Run the simulation
        """
        e_scores = []
        e_chromosomes = []
        pop_size = self.population.size()
        # pop_range = range(pop_size)
        eval_function = config["eval"]
        
        for ep in range(self.epochs):
            t = time.strftime("%X %x %Z")
            print(f"Generation: {ep+1} - {t}")

            future_envs = {
                self.executor.submit(sim_env, self.population.chromosomes[i]): (i, r)
                for i in range(pop_size)
                for r in range(self.runs)
            }
            for i, future in enumerate(as_completed(future_envs)):
                chrom_index, run = future_envs[future]
                if i % int(0.1 * pop_size * self.runs) == 0 and i != 0:
                    t = time.strftime("%X %x %Z")
                    print(f"Completed {i} chromosomes - {t}")
                try:
                    score = future.result()
                    self.scores[chrom_index][run] = score
                    # print(f"Chromosome {chrom_index}, run {run}: completed {score}")
                except Exception as e:
                    print(e)

            # Using executor.map, ignore
            # sim_args = [c for c in self.population.chromosomes for _ in range(self.runs)]
            # for chrom_index, score in zip(pop_range, self.executor.map(sim_env, sim_args, chunksize=16)):
            #     self.scores[chrom_index%self.population.size()] += score
            #     print(f"Chromosome {chrom_index%self.population.size()}: completed {score}")

            if eval_function == "median":
                self.population.scores = np.median(self.scores, axis=1)
            elif eval_function == "median_minvar":
                self.population.scores = np.median(self.scores, axis=1) - np.std(self.scores, axis=1)
            elif eval_function == "median_minvar_ratio":
                self.population.scores = np.median(self.scores, axis=1) / np.std(self.scores, axis=1)
            else:
                self.population.scores = np.min(self.scores, axis=1)
            self.population.makeBabies()

            best_index = np.argmax(self.population.scores)
            e_scores += [self.population.scores]
            best_score = e_scores[-1][best_index]
            print(
                f"Best {eval_function} score for epoch {ep+1}: {best_score} - chrom {best_index}"
            )
            #print(f"Time in thread: {time.thread_time()}\n")
            e_chromosomes += [self.population.chromosomes]
            
        #print(f"END Total Time: {time.thread_time()}\n")
        return (
            e_chromosomes,
            e_scores,
        )
