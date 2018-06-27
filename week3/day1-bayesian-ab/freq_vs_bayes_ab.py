#Frequentist A/B Approach

import pandas as pd
from scipy import stats
import numpy as np

# read in the data
df_a0 = pd.read_csv('data/a_2013-01-01.csv')

# find the existing CTR for a0
# model as a binomial distribution (series of bernoulli trials)
clicks_a0 = df_a0['converted'].sum() # number of successes, k
n_a0 = df_a0.shape[0] # total number of trials, n
p_a0 = clicks_a0 / n_a0 # probability of success, p
ctr_a0 = p_a0
print("The CTR of our present page is {0:0.4f}.".format(ctr_a0))

#We think we can do better! We have a new page, b that we think is at least 2% better (increase a from ~ 0.1 to 0.102).
#effect_size = 0.002
#H0: ctr_b - ctr_a < effect_size
#Ha: ctr_b - ctr_a >= effect_size
#this is a one-sided test

effect_size = 0.002 # desired effect size
alpha = 0.05 # desired significance level, also tolerance for Type I error (incorrectly rejecting the null, FP)
beta = 0.2 # desired tolerance for Type II error (failing to reject a false null hypotheses, FN)

delta = effect_size
p_a = 0.010 #probability of CTR a
p_b = 0.010 # without pilot study assume same
Z_alpha = stats.norm.ppf(1 - alpha) # one sided test
Z_beta = stats.norm.ppf(1 - beta)

#equation for sample size needed
n = int((Z_alpha + Z_beta)**2 / (-delta)**2*(p_a*(1-p_a) + p_b*(1-p_b)))

#collect samples
df_a = pd.read_csv('data/a_2013-01-02.csv', nrows=n) # only reading in nrows!
df_b = pd.read_csv('data/b_2013-01-02.csv', nrows=n)

a_clicks = df_a['converted'].sum() # they clicked, it was a success
ctr_a = a_clicks/n
b_clicks = df_b['converted'].sum()
ctr_b = b_clicks/n
#print("CTR a: {0:0.4f}, b: {1:0.4f}".format(ctr_a, ctr_b))

#two sample z-test
from statsmodels.stats.proportion import proportions_ztest
successes = [b_clicks, a_clicks]
nobs = [n, n] #number of observations
zstat, pval = proportions_ztest(successes,
                                nobs,
                                value=effect_size,
                                alternative='larger')
#print("The zstatistic is {0:0.3f}, with a p-value of {1:0.3e}.".format(zstat, pval))
#output: The zstatistic is 5.498, with a p-value of 1.922e-08.

#Bayesian

#1. Choose populations to compare
#Will be comparing page a data in data/a_2013-01-02.csv and page b data in data\b_2013-01-02.csv

#2. Choose distributions
#CTR is Binomial process = Beta distribution

#3. Set priors

# first make an x - array, goes 0 to 1 and will correspond to the ctr of site a and b
x = np.linspace(0, 1, num=1001, endpoint=True)

# setting prior - uninformative (each ctr could be from 0 to 1 with equal probability)
prior_a = stats.beta(a=1, b=1).pdf(x)
prior_b = stats.beta(a=1, b=1).pdf(x)

# for plotting
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
#
# import matplotlib
# matplotlib.rcParams.update({'font.size': 16})
#
# def plot_with_fill(x, y, label):
#     lines = plt.plot(x, y, label=label, lw=2)
#     plt.fill_between(x, 0, y, alpha=0.2, color=lines[0].get_c())
#     plt.legend(loc='best')
#
# plt.figure(figsize=(9,5))
# plot_with_fill(x, prior_a, 'a')
# plot_with_fill(x, prior_b, 'b')
# plt.xlabel('Click Through Rate')
# plt.ylabel('Frequency')
# plt.show()

#4. Gather data
n = 25 # read 50 rows of data
df_a = pd.read_csv('data/a_2013-01-02.csv', nrows=n)
df_b = pd.read_csv('data/b_2013-01-02.csv', nrows=n)

clicks_a = df_a['converted'].sum()
clicks_b = df_b['converted'].sum()
non_clicks_a = n - clicks_a
non_clicks_b = n - clicks_b
print("Of the first {0} rows, page a had {1} clicks, while b had {2} clicks.".format(n, clicks_a, clicks_b))

# recalculate priors and plot
alpha_a = clicks_a
beta_a = non_clicks_a
alpha_b = clicks_b
beta_b = non_clicks_b

prior_a = stats.beta(a=alpha_a, b=beta_a).pdf(x)
prior_b = stats.beta(a=alpha_b, b=beta_b).pdf(x)

plt.figure(figsize=(9,5))
plot_with_fill(x, prior_a, 'a')
plot_with_fill(x, prior_b, 'b')
plt.xlabel('Click Through Rate')
plt.ylabel('Frequency')
plt.title('After {0} data points'.format(n))
plt.show()

#5. Run simulation to see if B > A
sim_size = 10000
a_sample = np.random.beta(a=alpha_a, b=beta_a, size=sim_size)
b_sample = np.random.beta(a=alpha_b, b=beta_b, size=sim_size)

counts_b_larger = b_sample >= (a_sample + effect_size)
fraction_b_larger = counts_b_larger.sum()/sim_size
print(fraction_b_larger)
#output: .8152 - b is larger than effect size 80% of time

#Gather more data (repeat step 4)
n = 500 # read 500
df_a = pd.read_csv('data/a_2013-01-02.csv', nrows=n)
df_b = pd.read_csv('data/b_2013-01-02.csv', nrows=n)

clicks_a = df_a['converted'].sum()
clicks_b = df_b['converted'].sum()
non_clicks_a = n - clicks_a
non_clicks_b = n - clicks_b
print("Of the first {0} rows, page a had {1} clicks, while b had {2} clicks.".format(n, clicks_a, clicks_b))

# recalculate priors and plot
alpha_a = clicks_a
beta_a = non_clicks_a
alpha_b = clicks_b
beta_b = non_clicks_b

prior_a = stats.beta(a=alpha_a, b=beta_a).pdf(x)
prior_b = stats.beta(a=alpha_b, b=beta_b).pdf(x)

plt.figure(figsize=(9,5))
plot_with_fill(x, prior_a, 'a')
plot_with_fill(x, prior_b, 'b')
plt.xlabel('Click Through Rate')
plt.ylabel('Frequency')
plt.title('After {0} data points'.format(n))
plt.show()

#Simulate again
sim_size = 10000
a_sample = np.random.beta(a=alpha_a, b=beta_a, size=sim_size)
b_sample = np.random.beta(a=alpha_b, b=beta_b, size=sim_size)

counts_b_larger = b_sample >= (a_sample + effect_size)

fraction_b_larger = counts_b_larger.sum()/sim_size

print(fraction_b_larger)

#Gather more data, simulate again, etc.
