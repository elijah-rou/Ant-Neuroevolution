
import pickle
import numpy as np

pickle_off = open("results.pkl","rb")
emp = pickle.load(pickle_off)
# print(emp)
see = np.array(emp, dtype=object)
print(see.shape)
print(emp[0][0][0][0])