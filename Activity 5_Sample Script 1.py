# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:12:01 2024

@author: alexandria
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# set up linspace
t = np.linspace(0, 1, num=50)

# uniform distribution
uniform_dist = stats.uniform.pdf(t)
uniform_dist_normalized = uniform_dist / sum(uniform_dist)

# beta distribution
beta_dist = stats.beta.pdf(t, a=2, b=5)
beta_dist_normalized = beta_dist / sum(beta_dist)

# plot distributions
plt.plot(t, uniform_dist_normalized, 'r-', lw=1.5, label='Uniform Dist')
plt.plot(t, beta_dist_normalized, 'b-', lw=1.5, label='Beta Dist (a=2, b=5)')

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
