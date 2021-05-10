import sys
from ant_nn.simulation import Simulation
import dill


def main(config_path="config.yml", result_path="results.pkl"):
    sim = Simulation(config_path=config_path)
    chromosomes, scores, final_pop, food = sim.run()

    file = open(result_path, "wb")
    dill.dump([chromosomes, scores, final_pop, food], file)
    file.close()
    print("done")


if __name__ == '__main__':
    if len(sys.argv) > 2:
        main(config_path=sys.argv[1], result_path=sys.argv[2])
    elif len(sys.argv) > 1:
        main(config_path=sys.argv[1])
    else:
        main()