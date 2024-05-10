# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:18:25 2024

@author: alexandria
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

def likelihood_out (datum, mu):
    likelihood = sts.norm.pdf(datum, mu, scale = 0.1)
    return likelihood / likelihood.sum()

mu = np.linspace(start=1.65, stop=1.8, num=50)
likelihood_out_values = likelihood_out(1.7, mu)

plt.plot(mu, likelihood_out_values)
plt.title("Likelihood of $\\mu$ given observation 1.7m")
plt.xlabel('Value of $\\mu$')
plt.ylabel('Probability Density/Likelihood')
plt.show()
