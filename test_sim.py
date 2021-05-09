from ant_nn.simulation import Simulation
import dill


def main():
    sim = Simulation(config_path="short_config.yml")
    chromosomes, scores, food = sim.run()

    file = open("results.pkl", "wb")
    dill.dump([chromosomes, scores, food], file)
    file.close()
    print("done")


if __name__ == "__main__":
    main()
