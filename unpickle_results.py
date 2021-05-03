
import pickle

pickle_off = open("results.pkl","rb")
emp = pickle.load(pickle_off)
print(emp)