import math
import matplotlib.pyplot as plt
import numpy as np

sita = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
prior_of_coin_1 = np.array([1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11, 1 / 11])
prior_of_coin_2 = np.array([0.01, 0.01, 0.05, 0.08, 0.15, 0.4, 0.15, 0.08, 0.05, 0.01, 0.01])


def plot_bar(x_axis, y_axis, title_label=''):
    plt.ylim(0, 1)
    plt.bar(np.arange(len(y_axis)), y_axis)
    plt.xticks(np.arange(len(x_axis)), x_axis)
    plt.title(title_label)
    plt.show()


# Maximum likelihood Estimation in binomial distribution
def maximum_likelihood_estimation(prior, tossing_time=10, head=2):
    result_list = []
    for i in prior:
        result_list.append(
            (math.factorial(tossing_time) / (math.factorial(head) * math.factorial(tossing_time - head))) * (
                    i ** head) * ((1 - i) ** (tossing_time - head)))
    return np.array(result_list), np.argmax(result_list)


def maximum_posterior_estimation(prior, likelihood):
    posterior_list = (prior * likelihood) / sum(prior * likelihood)
    return np.array(posterior_list), np.argmax(posterior_list)


def tossing_myself(prior):
    # get random value between [0 ,1] ten times
    ten_coins_result = np.random.choice(2, 10)
    head_appear_time = np.count_nonzero(ten_coins_result == 1)
    # tail_appear_time = 10 - head_appear_time

    likelihood_list, _ = maximum_likelihood_estimation(sita, 10, head_appear_time)
    posterior_list, _ = maximum_posterior_estimation(prior, likelihood_list)
    return posterior_list


# hw (3)
# entropy
def calculate_entropy(collection_of_posterior):
    collection_list = list()
    for each_time_posterior in collection_of_posterior:
        sum = 0
        for i in each_time_posterior:
            if i != 0:
                sum += -(math.log(i, 2) * i)
        collection_list.append(sum)
    return collection_list


def fifty_times_tossing(first_prior):
    posterior = first_prior
    posterior_collection_list = list()
    for i in range(1, 51):
        posterior = tossing_myself(posterior)
        posterior_collection_list.append(posterior)
        if i % 10 == 0:
            plot_bar(sita, posterior, "observation_posterior_estimation " + str(i))
        # print(posterior)
    return posterior_collection_list


# hw (1)
# prior 1
print("---prior 1---")
plot_bar(sita, prior_of_coin_1, "prior 1")
likelihood_1, maximum_likelihood_of_sita_1 = maximum_likelihood_estimation(sita)
plot_bar(sita, likelihood_1, "likelihood_estimation 1")
print("maximum_likelihood sita: " + str(sita[maximum_likelihood_of_sita_1]))
posterior_1, maximum_posterior_of_sita_1 = maximum_posterior_estimation(prior_of_coin_1, likelihood_1)
plot_bar(sita, posterior_1, "posterior_estimation 1")
print("maximum_posterior sita: " + str(sita[maximum_posterior_of_sita_1]))

# prior 2
print("---prior 2---")
plot_bar(sita, prior_of_coin_2, "prior 2")
likelihood_2, maximum_likelihood_of_sita_2 = maximum_likelihood_estimation(sita)
plot_bar(sita, likelihood_2, "likelihood_estimation 2")
print("maximum_likelihood sita: " + str(sita[maximum_likelihood_of_sita_2]))
posterior_2, maximum_posterior_of_sita_2 = maximum_posterior_estimation(prior_of_coin_2, likelihood_2)
plot_bar(sita, posterior_2, "posterior_estimation 2")
print("maximum_posterior sita: " + str(sita[maximum_posterior_of_sita_2]))

# hw (2)
# requirement: tossing 50 times, observation every ten times
# prior 1
print("---prior 1---")
pi1 = fifty_times_tossing(prior_of_coin_1)
# prior 2
print("---prior 2---")
pi2 = fifty_times_tossing(prior_of_coin_2)

# hw (3)
a = calculate_entropy(pi1)
plt.plot(a)
plt.title("prior 1 entropy")
plt.show()
b = calculate_entropy(pi2)
plt.plot(b)
plt.title("prior 2 entropy")
plt.show()
