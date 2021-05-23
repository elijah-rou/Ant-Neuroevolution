from ant_nn.agent.DiscretAnt2 import DiscretAnt2
import numpy as np
import matplotlib.pyplot as plt
from ant_nn.environ.Environment import Environment
from ant_nn.agent.population import Population
import yaml
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from torch.optim import Adam


def sim_env(timesteps, config, chromosome=None, model=None):
    assert (chromosome is not None) or (model is not None)
    sim = {"env": Environment(config, chromosome), "food": np.zeros(timesteps)}
    for t in range(timesteps):
        sim["env"].update()
        sim["food"][t] = sim["env"].nest.food
    score = sim["food"]
    return score

def sim_env_backprop(timesteps, config, model):
    assert (model is not None)
    sim = {"env": Environment(config, True, model=model), "food": np.zeros(timesteps)}
    for t in range(timesteps):
        sim["env"].update()
        sim["food"][t] = sim["env"].nest.food
    score = sim["food"]
    return score, sim["env"].get_agent_scores()

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
        self.config = yaml.full_load(file_stream)

        ga_config = self.config["population"]

        self.eval_function = self.config["eval"]
        self.timesteps = self.config["num_timesteps"]
        self.epochs = self.config["num_epochs"]
        self.runs = self.config["num_runs"]
        self.population = Population(
            ga_config["size"],
            ga_config["mutation_rate"],
            ga_config["crossover_rate"],
            ga_config["crossover_flag"],
            ga_config["mutation_strength"],
            ga_config["keep_threshold"],
            self.config["agent"],
            ga_config["init_from_file"],
            ga_config["filename"]
        )

        self.executor = ProcessPoolExecutor()
        self.scores = np.zeros((self.population.size(), self.runs))
        self.food_res = np.zeros((self.population.size(), self.runs, self.timesteps))
        self.config = self.config

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
    
    def run_backprop(self):
        agent_config = self.config["agent"]
        params = agent_config.get("params")
        layer_size = params.get("hidden_layer_size")
        height = self.config.get("height", 50)
        width = self.config.get("width", 50)
        if isinstance(self.config["nest_location"], str):
            if self.config["nest_location"] == "center":
                nest_loc = [height // 2, width // 2]
            elif self.config["nest_location"] == "origin":
                nest_loc = [0, 0]
        else:
            nest_loc = self.config["nest_location"]

        if agent_config["type"] == "DiscretAnt2":
            d_bins = params.get("direction_bins", 7)
            model = DiscretAnt2(layer_size, None, d_bins, nest_loc=nest_loc, position=nest_loc).brain

        # NN Params
        gamma = 0.99
        lr = 0.01
        optimizer = Adam(model.parameters(), lr)

        for ep in range(self.epochs):
            t = time.strftime("%X %x %Z")
            print(f"Generation: {ep+1} - {t}")
            food, agent_scores = sim_env_backprop(self.timesteps, self.config, model)
            agent_scores = agent_scores
            discounted = torch.zeros((self.config["num_agents"], self.timesteps))
            agent_losses = np.zeros(self.config["num_agents"])
            for agent in range(discounted.shape[0]):
                for t in range(discounted.shape[1]):
                    Gt = 0
                    power = 0
                    for reward in agent_scores[agent][0][t:]:
                        Gt += gamma**power * reward
                        power += 1
                    discounted[agent, t] = Gt
                
                discounted[agent] = (discounted[agent] - torch.mean(discounted[agent]))/torch.std(discounted[agent])
                loss = -torch.stack(agent_scores[agent][1])*discounted[agent]
                optimizer.zero_grad()
                loss = loss.sum()
                agent_losses[agent] = loss.item()
                loss.backward()
                optimizer.step()
                print(f"Agent {agent} loss: {loss.item()}")
            print(f"Mean Loss: {np.mean(agent_losses)}")
            print(f"Food: {food[-1]}")


