import random
import math
import matplotlib.pyplot as plt
import numpy as np

sita = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
prior_of_coin_1 = np.array([1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11])
prior_of_coin_2 = np.array([0.01, 0.01, 0.05, 0.08, 0.15, 0.4, 0.15, 0.08, 0.05, 0.01, 0.01])


def polt_bar(x_axis, y_axis, title_label):
    plt.ylim(0, 1)
    plt.bar(np.arange(len(y_axis)), y_axis)
    plt.xticks(np.arange(len(x_axis)), x_axis)
    plt.title(title_label)
    plt.show()


# Maximum likelihood Estimation in binomial distribution
def maximum_likelihood_estimation(prior, tossing_time=10, head=2):
    result_list = []
    for i in prior:  # tossing ten times
        result_list.append(
            (math.factorial(tossing_time) / (math.factorial(head) * math.factorial(tossing_time - head))) * (
                    i ** head) * ((1 - i) ** (tossing_time - head)))
        # current_p = prior_of_coin_1[random.randint(1, 11) - 1]
        # property_result += (math.factorial(n) / (math.factorial(i) * math.factorial(n - i))) * current_p
    return np.array(result_list), np.argmax(result_list)


def maximum_posterior_estimation(prior, likelihood):
    posterior_list = (prior * likelihood) / sum(prior * likelihood)
    return posterior_list, np.argmax(posterior_list)


def tossing_myself(prior):
    # get random value between [0 ,1] ten times
    ten_coins_result = np.random.choice(2, 10)
    head_appear_time = np.count_nonzero(ten_coins_result == 1)
    # tail_appear_time = 10 - head_appear_time

    likelihood_list, _ = maximum_likelihood_estimation(prior, 10, head_appear_time)
    posterior, _ = maximum_posterior_estimation(prior, likelihood_list)
    print("***")
    print(likelihood_list)
    print("*")
    print(posterior)
    return posterior


def fifty_times_tossing(first_prior):
    posterior = first_prior
    for i in range(1, 6):
        posterior = tossing_myself(posterior)
        polt_bar(sita, posterior, "observation_posterior_estimation " + str(i))
        # print(posterior)


# prior 1
print("---prior 1---")
polt_bar(sita, prior_of_coin_1, "prior 1")
likelihood_1, maximum_likelihood_of_sita_1 = maximum_likelihood_estimation(sita)
polt_bar(sita, likelihood_1, "likelihood_estimation 1")
print("maximum_likelihood sita: " + str(sita[maximum_likelihood_of_sita_1]))
posterior_1, maximum_posterior_of_sita_1 = maximum_posterior_estimation(prior_of_coin_1, likelihood_1)
polt_bar(sita, posterior_1, "posterior_estimation 1")
print("maximum_posterior sita: " + str(sita[maximum_posterior_of_sita_1]))

# prior 2
print("---prior 2---")
polt_bar(sita, prior_of_coin_2, "prior 2")
likelihood_2, maximum_likelihood_of_sita_2 = maximum_likelihood_estimation(sita)
polt_bar(sita, likelihood_2, "likelihood_estimation 2")
print("maximum_likelihood sita: " + str(sita[maximum_likelihood_of_sita_2]))
posterior_2, maximum_posterior_of_sita_2 = maximum_posterior_estimation(prior_of_coin_2, likelihood_2)
polt_bar(sita, posterior_2, "posterior_estimation 1")
print("maximum_posterior sita: " + str(sita[maximum_posterior_of_sita_2]))

fifty_times_tossing(prior_of_coin_1)
