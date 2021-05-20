import matplotlib.pyplot as plt
import numpy as np
import pickle
from ant_nn.plot import *

result = pickle.load(open("./results.pkl", "rb"))

plot_food_over_time(result)

# fetchant_1_temp = pickle.load(open("./fetchant_1.pkl", "rb"))
# fetchant_2_temp = pickle.load(open("./fetchant_2.pkl", "rb"))

# fetchant_1 = {
#     "chromosomes" : fetchant_1_temp[0],
#     "scores" : np.asarray(fetchant_1_temp[1])
# }

# fetchant_2 = {
#     "chromosomes" : fetchant_2_temp[0],
#     "scores" : np.asarray(fetchant_2_temp[1])
# }

# plot_score_evolution(fetchant_2["scores"])
# plot_score_evolution(fetchant_1["scores"])

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def exp_decay(x):
#     return (1 - np.exp(-x))

# x = np.array(range(400))/40
# fake_scores = 20*sigmoid(x) + np.random.normal(0,0.5,400)

# # fake score evolution
# plt.plot(fake_scores)
# plt.xlabel('epoch')
# plt.ylabel('score')
# plt.show()

# # fake food consumed comparrison
# x = np.array(range(400))/20
# fake_food1 = exp_decay(x)
# fake_food2 = 3*exp_decay(x)
# fake_food3 = 60*exp_decay(x/5)
# fake_food4 = 60*exp_decay(x)
# plt.plot(fake_food1)
# plt.plot(fake_food2)
# plt.plot(fake_food3)
# plt.plot(fake_food4)
# plt.xlabel('time step')
# plt.ylabel('food gathered')
# plt.legend(['epoch 1', 'epoch 20', 'epoch 300', 'epoch 500'])
<<<<<<< HEAD
# plt.show()
=======
# plt.show()
>>>>>>> 914c54ea8104d26f390bf563b2399a8165c3c93c
