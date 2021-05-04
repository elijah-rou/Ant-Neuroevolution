
import pickle
import numpy as np

pickle_off = open("results.pkl","rb")
temp = pickle.load(pickle_off)
print(temp)
level = temp[1][-1]
np.array(level)
print(type(level))
print(len(level))
