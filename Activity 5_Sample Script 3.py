# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:25:13 2024

@author: alexandria
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

# Define the likelihood function
def likelihood(mu, datum, scale):
    return sts.norm.pdf(datum, mu, scale)

# Given observation
datum = 1.7

# Generate a range of values for mu
mu_values = np.linspace(start=-1, stop=1.8, num=50)

# Compute the likelihood for these values given the observation
likelihood_out = [likelihood(mu, datum, scale=0.1) for mu in mu_values]

# Define a uniform distribution
uniform_dist = np.ones_like(mu_values)

# Multiplying the likelihood by the prior (uniform distribution in this case)
unnormalized_posterior = likelihood_out * uniform_dist

# Plotting the unnormalized posterior distribution
plt.plot(mu_values, unnormalized_posterior)
plt.title("Unnormalized Posterior")
plt.xlabel('$\mu$ in meters')
plt.ylabel('Unnormalized Posterior')
plt.show()
