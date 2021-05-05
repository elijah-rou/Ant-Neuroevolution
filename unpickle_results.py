import pickle
import numpy as np

pickle_off = open("results.pkl","rb")
temp = pickle.load(pickle_off)
print(temp)
level = temp[1][-1]
np.array(level)
print(type(level))
print(len(level))

def get_best(emp):
    best_index = np.argmax(np.max(emp[1], axis=1))
    return emp[0][best_index][np.argmax(emp[1], axis=1)[best_index]]
