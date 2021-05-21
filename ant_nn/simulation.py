import numpy as np
import matplotlib.pyplot as plt
from ant_nn.environ.Environment import Environment
from ant_nn.agent.population import Population
import yaml
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def sim_env(timesteps, config, chromosome=None, model=None):
    assert (chromosome is not None) or (model is not None)
    sim = {"env": Environment(config, chromosome), "food": np.zeros(timesteps)}
    for t in range(timesteps):
        sim["env"].update()
        sim["food"][t] = sim["env"].nest.food
    score = sim["food"]
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
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        file_stream = open(config_path, "r")
        config = yaml.full_load(file_stream)

        ga_config = config["population"]
        agent_params = config["agent"]["params"]

        self.eval_function = config["eval"]
        self.timesteps = config["num_timesteps"]
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
            ga_config["init_from_file"],
            ga_config["filename"],
        )

        self.executor = ProcessPoolExecutor()
        self.scores = np.zeros((self.population.size(), self.runs))
        self.food_res = np.zeros((self.population.size(), self.runs, self.timesteps))
        self.config = config

    def run(self, degen_epoch=None, degen_score=10):
        """
        Run the simulation
        """
        
        pop_is_degen = True  # used to restart sim if population isn't evolving
        while (pop_is_degen):
            e_chromosomes = []
            final_pop = []
            pop_size = self.population.size()
            e_scores = np.zeros((self.epochs, pop_size))
            max_score = 0
            # pop_range = range(pop_size)
            # eval_function = config["eval"]

            for ep in range(self.epochs):
                t = time.strftime("%X %x %Z")
                print(f"Generation: {ep+1} - {t}")

                future_envs = {
                    self.executor.submit(
                        sim_env, self.timesteps, self.config, self.population.chromosomes[i]
                    ): (i, r)
                    for i in range(pop_size)
                    for r in range(self.runs)
                }
                for i, future in enumerate(as_completed(future_envs)):
                    chrom_index, run = future_envs[future]
                    if i % int(0.1 * pop_size * self.runs) == 0 and i != 0:
                        t = time.strftime("%X %x %Z")
                        print(f"Completed {i} chromosomes - {t}")
                    try:
                        food_results = future.result()
                        self.food_res[chrom_index][run] = food_results
                        self.scores[chrom_index][run] = food_results[-1]
                        # print(f"Chromosome {chrom_index}, run {run}: completed {score}")
                    except Exception as e:
                        print(e)

                # Using executor.map, ignore
                # sim_args = [c for c in self.population.chromosomes for _ in range(self.runs)]
                # for chrom_index, score in zip(pop_range, self.executor.map(sim_env, sim_args, chunksize=16)):
                #     self.scores[chrom_index%self.population.size()] += score
                #     print(f"Chromosome {chrom_index%self.population.size()}: completed {score}")

                if self.eval_function == "median":
                    self.population.scores = np.median(self.scores, axis=1)
                elif self.eval_function == "median_minvar":
                    self.population.scores = np.median(self.scores, axis=1) - np.std(
                        self.scores, axis=1
                    )
                elif self.eval_function == "median_minvar_ratio":
                    self.population.scores = np.median(self.scores, axis=1) / np.std(
                        self.scores, axis=1
                    )
                elif self.eval_function == "average":
                    self.population.scores = np.mean(self.scores, axis=1)
                else:
                    self.population.scores = np.min(self.scores, axis=1)
                self.population.makeBabies()

                best_index = np.argmax(self.population.scores)
                e_scores[ep] = self.population.scores
                best_score = e_scores[ep][best_index]
                med_score = np.median(e_scores[ep])
                print(
                    f"Best {self.eval_function} score for epoch {ep+1}: {best_score} - chrom {best_index}"
                )
                print(
                    f"Median {self.eval_function} score for epoch {ep+1}: {med_score}\n"
                )
                # print(f"Time in thread: {time.thread_time()}\n")
                e_chromosomes += [self.population.chromosomes[best_index]]

                max_score = max(max_score,best_score)  # tracks max score over all epochs
                if (degen_epoch is not None):  # if degen resim is enabled
                    if (ep+1 >= degen_epoch):  # if past degen_epoch
                        if (max_score < degen_score):  # if max score across all epochs < degen_score
                            print('DEGENERATE: degen_epoch reached before degen_score. Restarting... \n')
                            is_degen = True  # restart the sim
                            break
                else:
                    is_degen = False

        final_pop = self.population.chromosomes

        # print(f"END Total Time: {time.thread_time()}\n")
        return (e_chromosomes, e_scores, final_pop, self.food_res)


class BackpropSimulation:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        file_stream = open(config_path, "r")
        config = yaml.full_load(file_stream)

        self.eval_function = config["eval"]
        self.timesteps = config["num_timesteps"]
        self.epochs = config["num_epochs"]
        self.runs = config["num_runs"]

        self.executor = ProcessPoolExecutor()
        self.scores = np.zeros((self.population.size(), self.runs))
        self.food_res = np.zeros((self.population.size(), self.runs, self.timesteps))
        self.config = config

    def run(self, degen_epoch=None, degen_score=10):
        """
        Run the simulation
        """

        for ep in range(self.epochs):
            t = time.strftime("%X %x %Z")
            print(f"Generation: {ep+1} - {t}")

            score = sim_env(self.timesteps, self.config, model=model)

            # Using executor.map, ignore
            # sim_args = [c for c in self.population.chromosomes for _ in range(self.runs)]
            # for chrom_index, score in zip(pop_range, self.executor.map(sim_env, sim_args, chunksize=16)):
            #     self.scores[chrom_index%self.population.size()] += score
            #     print(f"Chromosome {chrom_index%self.population.size()}: completed {score}")

            if self.eval_function == "REINFORCE":
                self.population.scores = np.median(self.scores, axis=1)

            best_index = np.argmax(self.population.scores)
            e_scores[ep] = self.population.scores
            best_score = e_scores[ep][best_index]
            med_score = np.median(e_scores[ep])
            print(
                f"Best {self.eval_function} score for epoch {ep+1}: {best_score} - chrom {best_index}"
            )
            print(
                f"Median {self.eval_function} score for epoch {ep+1}: {med_score}\n"
            )
            # print(f"Time in thread: {time.thread_time()}\n")
            e_chromosomes += [self.population.chromosomes[best_index]]

            max_score = max(max_score,best_score)  # tracks max score over all epochs
            if (degen_epoch is not None):  # if degen resim is enabled
                if (ep+1 >= degen_epoch):  # if past degen_epoch
                    if (max_score < degen_score):  # if max score across all epochs < degen_score
                        print('DEGENERATE: degen_epoch reached before degen_score. Restarting... \n')
                        is_degen = True  # restart the sim
                        break
            else:
                is_degen = False

        final_pop = self.population.chromosomes

        # print(f"END Total Time: {time.thread_time()}\n")
        return (e_chromosomes, e_scores, final_pop, self.food_res)
