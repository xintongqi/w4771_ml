import numpy as np
import matplotlib.pyplot as plt

# Pr(Y=1|X=x) = p0 if x <= t1
# Pr(Y=1|X=x) = p1 if t1 < x < t2
# Pr(Y=1|X=x) = p2 if x >= t2

t1 = 0.2
t2 = 0.8
p0 = 0.25
p1 = 0.6
p2 = 0.3

################################################################################
# Part (b)

# Function to implement
def stump_err(t, a, b):
    # XXX imlement me
    return 0

print(f'T1: {stump_err(0.4,0,1)}')
print(f'T2: {stump_err(0.5,0,1)}')
print(f'T3: {stump_err(0.5,1,0)}')

################################################################################
# Part (c)

# Plotting code
t_vals = np.linspace(-0.2, 1.2, num=14001)
best_stump_err_rates = [] # XXX implement me

plt.figure()
plt.plot(t_vals, best_stump_err_rates)
plt.xlabel('$t$')
plt.ylabel('best stump error rate with predicate $x \leq t$')
plt.savefig('error_rates.pdf', bbox_inches='tight')
plt.close()

################################################################################
# Part (d)

# Function to implement
def find_best_stump(x,y):
    # XXX implement me
    t = 0
    a = 0
    b = 0
    train_err_rate
    return (t, a, b, train_err_rate)

x = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9 ])
y = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1])
t, a, b, train_err = find_best_stump(x, y)
print(f'{(t,a,b)}, {train_err}')

################################################################################
# Part (e)

# Function to generate data
def generate_data(n):
    x = np.random.rand(n)
    z = np.random.rand(n)
    y = np.zeros(n)
    y[x <= t1] = z[x <= t1] <= p0
    y[(x > t1) * (x < t2)] = z[(x > t1) * (x < t2)] <= p1
    y[x >= t2] = z[x >= t2] <= p2
    return (x,y)

# Simulation code
np.random.seed(42)
n = 100
num_trials = 5000
error_rates = np.zeros(num_trials)
thresholds = np.zeros(num_trials)
for trial in range(num_trials):
    t, a, b, _ = find_best_stump(*generate_data(n))
    thresholds[trial] = t
    error_rates[trial] = stump_err(t, a, b)

# Plotting code
plt.figure()
plt.hist(thresholds, bins=50)
plt.xlabel('$\hat\\theta$')
plt.ylabel('counts')
plt.savefig('histogram1.pdf', bbox_inches='tight')
plt.close()

plt.figure()
plt.hist(error_rates, bins=50)
plt.xlabel('error rate')
plt.ylabel('counts')
plt.savefig('histogram2.pdf', bbox_inches='tight')
plt.close()

