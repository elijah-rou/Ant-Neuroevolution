from ant_nn.simulation import Simulation
import dill


def main():
    sim = Simulation(config_path="config.yaml")
    chromosomes, scores, food, final_pop = sim.run()

    file = open("results.pkl", "wb")
    dill.dump([chromosomes, scores, food, final_pop], file)
    file.close()
    print("done")


if __name__ == "__main__":
    main()
