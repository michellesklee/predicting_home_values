import numpy as np
import pandas as pd
import scipy.stats as scs
import matplotlib.pyplot as plt

n = 800
dfA = pd.read_csv('~/galvanize/week3/day1-bayesian-ab/dsi-multi-armed-bandit/data/siteA.txt', nrows=n)
dfB = pd.read_csv('~/galvanize/week3/day1-bayesian-ab/dsi-multi-armed-bandit/data/siteB.txt', nrows=n)

x = np.arange(0, 1.01, 0.01)

clicks_a =  dfA['0'].sum()
clicks_b = dfB['0'].sum()
non_clicks_a = n - clicks_a
non_clicks_b = n - clicks_b
#print("Of the first {0} rows, page a had {1} clicks, while b had {2} clicks.".format(n, clicks_a, clicks_b))

alpha_a = clicks_a
beta_a = non_clicks_a
alpha_b = clicks_b
beta_b = non_clicks_b

#uniform priors
# prior_a = scs.beta(a=1, b=1).pdf(x)
# prior_b = scs.beta(a=1, b=1).pdf(x)

#priors after sample
prior_a = scs.beta(a=alpha_a, b=beta_a).pdf(x)
prior_b = scs.beta(a=alpha_b, b=beta_b).pdf(x)

# Uniform distribution with just site A
# x = np.arange(0, 1.01, 0.01)
# y = scs.uniform().pdf(x)
#
# def plot_with_fill(x, y, label):
#     lines = plt.plot(x, y, label=label, lw=2)
#     plt.fill_between(x, 0, y, alpha=0.2, color=lines[0].get_c())
#     plt.legend(loc='best')
# plt.figure(figsize=(9,5))
# plot_with_fill(x, y, 'Prior')
# plt.xlabel('Click Through Rate')
# plt.ylabel('Frequency')
# plt.show()

#Beta distribution for A
# x = np.arange(0, 1.01, 0.01)
# y = scs.beta(a=1, b=1).pdf(x)
#
# def plot_with_fill(x, y, label):
#     lines = plt.plot(x, y, label=label, lw=2)
#     plt.fill_between(x, 0, y, alpha=0.2, color=lines[0].get_c())
#     plt.legend(loc='best')
# plt.figure(figsize=(9,5))
# plot_with_fill(x, y, 'Prior')
# plot_with_fill(x, prior_a, 'Posterior after 50 views')
# plt.xlabel('Click Through Rate')
# plt.ylabel('Frequency')
# plt.show()
# plt.savefig('siteA_50views.png')
# plt.savefig('siteA_150views.png')

#Beta distribution for A and B
def plot_with_fill(x, y, label):
    lines = plt.plot(x, y, label=label, lw=2)
    plt.fill_between(x, 0, y, alpha=0.2, color=lines[0].get_c())
    plt.legend(loc='best')
# plt.figure(figsize=(9,5))
# plot_with_fill(x, prior_a, 'a')
# plot_with_fill(x, prior_b, 'b')
# plt.xlabel('Click Through Rate')
# plt.ylabel('Frequency')
# plt.show()
# plt.savefig('siteA_B_800views.png')

sim_size = 10000
a_sample = np.random.beta(a=alpha_a, b=beta_a, size=sim_size)
b_sample = np.random.beta(a=alpha_b, b=beta_b, size=sim_size)

counts_b_larger = b_sample > a_sample
fraction_b_larger = (counts_b_larger.sum()) / 10000
#print(fraction_b_larger)
#output: 0.9952

#95% credible interval
#for site A
lower_a = scs.beta(a=clicks_a, b=non_clicks_a).ppf(0.025)
upper_a = scs.beta(a=clicks_a, b=non_clicks_a).ppf(0.975)
#print("A's 95% HDI is {:.5f} to {:.5f}".format(lower_a, upper_a))
#for site B
lower_b = scs.beta(a=clicks_b, b=non_clicks_b).ppf(0.025)
upper_b = scs.beta(a=clicks_b, b=non_clicks_b).ppf(0.975)
#print("B's 95% HDI is {:.5f} to {:.5f}".format(lower_b, upper_b))

#10. 
