import pickle
import numpy as np

def get_best(results):
    best_index = np.argmax(np.max(results[1], axis=1))
    return results[0][best_index]#[np.argmax(results[1], axis=1)[best_index]]

if __name__ == "__main__":
    pickle_off = open("results.pkl", "rb")
    temp = pickle.load(pickle_off)
    # print(temp)
    level = temp[1][-1]
    np.array(level)
    print(type(level))
    print(len(level))