# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:36:18 2024

@author: alexandria
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simulate_linear_data(start, stop, N, beta_0, beta_1, eps_mean, eps_sigma_sq):
    """
    Simulate a random dataset using a noisy
    linear process.

    Parameters
    ----------
    N: `int`
        Number of data points to simulate
    beta_0: `float`
        Intercept
    beta_1: `float`
        Slope of univariate predictor, X

    Returns
    -------
    df: `pd.DataFrame`
        A DataFrame containing the x and y values.
    """
    # Create a pandas DataFrame with column 'x' containing
    # N uniformly sampled values between `start` and `stop`
    df = pd.DataFrame({"x": np.linspace(start, stop, num=N)})

    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to 
    # generate a column 'y' of responses based on 'x'
    df["y"] = beta_0 + beta_1 * df["x"] + np.random.RandomState(42).normal(eps_mean, eps_sigma_sq, N)
    
    return df


def plot_simulated_data(df):
    """
    Plot the simulated data with a regression line.

    Parameters
    ----------
    df: `pd.DataFrame`
        A DataFrame containing the x and y values.
    """
    # Scatter plot of the data
    plt.scatter(df["x"], df["y"])

    # Calculate regression line using polyfit
    coeffs = np.polyfit(df["x"], df["y"], deg=1)
    regression_line = coeffs[0] * df["x"] + coeffs[1]

    # Plot the regression line
    plt.plot(df["x"], regression_line, color='blue')

    # Remove plot title
    plt.gca().set_title('')

    # Remove legend
    plt.gca().legend([])

    # Show plot
    plt.show()


if __name__ == "__main__":
    # True parameters
    beta_0 = 1.0  # Intercept
    beta_1 = 2.0  # Slope

    # Simulate data
    start = 0
    stop = 1
    N = 100
    eps_mean = 0.0
    eps_sigma_sq = 0.5
    df = simulate_linear_data(start, stop, N, beta_0, beta_1, eps_mean, eps_sigma_sq)

    # Plot simulated data with regression line
    plot_simulated_data(df)
    
