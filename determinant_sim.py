# import matplotlib.pyplot as plt
import numpy as np
from ant_nn.environ.Environment import Environment


def main():
    sm = 0
    env = Environment(config_path="determinant_config.yaml")
    food_retrived = env.run(max_t=300)
    # np.save("food_retrieved", food_retrived)
    print(food_retrived[-1])
    # plt.plot(food_retrived)


if __name__ == "__main__":
    main()
