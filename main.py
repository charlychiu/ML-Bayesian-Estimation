import random
import math
import matplotlib.pyplot as plt
import numpy as np

sita = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
prior_of_coin_1 = np.array([1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11])
prior_of_coin_2 = np.array([0.01, 0.01, 0.05, 0.08, 0.15, 0.4, 0.15, 0.08, 0.05, 0.01, 0.01])

def polt_bar(x_axis, y_axis):
    plt.ylim(0, 1)
    plt.bar(np.arange(len(prior_of_coin_1)), prior_of_coin_1)
    plt.xticks(np.arange(len(sita)), sita)
    plt.show()
    plt.ylim(0, 1)
    plt.bar(np.arange(len(prior_of_coin_2)), prior_of_coin_2)
    plt.xticks(np.arange(len(sita)), sita)
    plt.show()




polt_bar(sita, prior_of_coin_1)
polt_bar(sita, prior_of_coin_1)
# Maximum likelihood Estimation in binomial distribution
# n = 10  # tossing count
# head_count = 2
# property_result = 0
# for i in range(1, 11):  # tossing ten times
#     current_p = prior_of_coin_1[random.randint(1, 11) - 1]
#     property_result += (math.factorial(n) / (math.factorial(i) * math.factorial(n - i))) * current_p
#
# # print(property_result)

# plot prior

