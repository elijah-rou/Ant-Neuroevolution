import matplotlib.pyplot as plt
import numpy as np
import pickle as pk


def res_to_dict(result):
    res_dict = {
        "chromosomes": result[0],
        "scores": np.asarray(result[1]),
        "food": np.asarray(result[2]),
    }
    return res_dict


def plot_score_evolution(result):
    """plots best score vs epoch"""

    res_dict = res_to_dict(result)
    best_scores = np.max(res_dict["scores"], axis=1)

    plt.plot(best_scores)
    plt.xlabel("epoch")
    plt.ylabel("best score")
    plt.title("Score Evolution")
    plt.show()


def plot_food_over_time(result, num_epochs=4):
    """plots best food_gathered vs time for each epoch"""

    res_dict = res_to_dict(result)
    food = res_dict["food"]
    epochs = [(i + 1) * len(food) // num_epochs for i in range(num_epochs)]

    for e in epochs:
        plt.plot(food[e - 1, 0, :])
    plt.legend(epochs)
    plt.show()
    pass
