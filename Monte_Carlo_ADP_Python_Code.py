# The code below takes a CSV data file and tries to fit a beta distribution function for every interval of the 24 hour day.

import pandas as pd
import numpy as np
from scipy.stats import beta, probplot
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib.pyplot as plt


# Fix bugs and improve PDF fitting as needed

def fit_beta_distribution(filename, show_plots):
    # Import Data
    df = read_data_from_file(filename)
    if df is None:
        return

    # Generate hypothetical interval names for reference.
    interval_names = [f"Interval_{i}" for i in range(1, df.shape[1] + 1)]
    
    # Assigning names to DataFrame columns
    df.columns = interval_names
    
    # Creating a dictionary to store distribution parameters for each interval
    interval_distributions = {}
    interval_pdf_functions = {}

    # Define a small epsilon to ensure data is strictly within (0, 1)
    epsilon = 1e-5
    
    # Iterating through each interval
    for interval in interval_names:
        # Extract the data for the interval across all days
        interval_data = df[interval]
        
        if not np.isfinite(interval_data).all():
            print(f"Warning: {interval} in {filename} contains non-finite values.")
            interval_data = interval_data.dropna()  # drop NaNs
            interval_data = interval_data[np.isfinite(interval_data)]  # drop infinite values

        # Perform scaling if necessary (ensure data is between 0 and 1)
        interval_data_scaled = (interval_data - interval_data.min()) / (interval_data.max() - interval_data.min())
        
        # Adjust scaled data to be strictly within (0, 1)
        interval_data_scaled_adj = np.clip(interval_data_scaled, epsilon, 1-epsilon)
        
        # Fitting a beta distribution to the adjusted interval data
        a, b, loc, scale = beta.fit(interval_data_scaled_adj, floc=0, fscale=1)
        
        # Storing the distribution parameters
        interval_distributions[interval] = (a, b, loc, scale)

        # Store the PDF function for this interval
        interval_pdf_functions[interval] = lambda x, a=a, b=b, loc=loc, scale=scale: beta.pdf(x, a, b, loc, scale)

        # Visualize the original and fitted data
        x = np.linspace(epsilon, 1-epsilon, 1000)
        pdf_values = beta.pdf(x, a, b, loc, scale)
        
        if show_plots:
            plt.figure(figsize=(8, 6))
            plt.hist(interval_data_scaled_adj, bins=15, density=True, alpha=0.6, color='g')
            plt.plot(x, pdf_values, linewidth=2, color='k')
            plt.title(f'Beta Distribution Fit for {interval}')
            plt.xlabel('Scaled and Adjusted Data')
            plt.ylabel('Density')
            plt.show()
            
            # Creating Q-Q plot
            plt.figure(figsize=(8, 6))
            probplot(interval_data_scaled_adj, dist=beta, sparams=(a, b, loc, scale), plot=plt)
            plt.title(f'Q-Q Plot for {interval} (Beta distribution)')
            plt.xlabel('Theoretical quantiles')
            plt.ylabel('Ordered Values')
            plt.grid(True)
            plt.show()

    return interval_distributions, interval_pdf_functions

def fit_gamma_distribution(filename, show_plots):
    # Import Data
    df = read_data_from_file(filename)
    if df is None:
        return

    # Generate hypothetical interval names for reference.
    interval_names = [f"Interval_{i}" for i in range(1, df.shape[1] + 1)]
    
    # Assigning names to DataFrame columns
    df.columns = interval_names
    
    # Creating a dictionary to store distribution parameters for each interval
    interval_distributions = {}
    interval_pdf_functions = {}
    
    # Iterating through each interval
    for interval in interval_names:
        # Extract the data for the interval across all days
        interval_data = df[interval]
        
        if not np.isfinite(interval_data).all():
            print(f"Warning: {interval} in {filename} contains non-finite values.")
            interval_data = interval_data.dropna()  # drop NaNs
            interval_data = interval_data[np.isfinite(interval_data)]  # drop infinite values
        
        # Fitting a gamma distribution to the data
        a, loc, scale = gamma.fit(interval_data)
        
        # Storing the distribution parameters
        interval_distributions[interval] = (a, loc, scale)

        # Store the PDF function for this interval
        interval_pdf_functions[interval] = lambda x, a=a, loc=loc, scale=scale: gamma.pdf(x, a, loc, scale)
        
        # Visualize the original and fitted data
        x = np.linspace(min(interval_data), max(interval_data), 1000)
        pdf_values = gamma.pdf(x, a, loc, scale)
        
        if show_plots:
            plt.figure(figsize=(8, 6))
            plt.hist(interval_data, bins=15, density=True, alpha=0.6, color='g')
            plt.plot(x, pdf_values, linewidth=2, color='k')
            plt.title(f'Gamma Distribution Fit for {interval}')
            plt.xlabel('Data')
            plt.ylabel('Density')
            plt.show()
            
            # Creating Q-Q plot
            plt.figure(figsize=(8, 6))
            probplot(interval_data, dist=gamma, sparams=(a, loc, scale), plot=plt)
            plt.title(f'Q-Q Plot for {interval} (Gamma distribution)')
            plt.xlabel('Theoretical quantiles')
            plt.ylabel('Ordered Values')
            plt.grid(True)
            plt.show()

    return interval_distributions, interval_pdf_functions

def fit_gaussian_distribution(filename, show_plots):
    # Import Data
    df = read_data_from_file(filename)
    if df is None:
        return

    # Generate hypothetical interval names for reference.
    interval_names = [f"Interval_{i}" for i in range(1, df.shape[1] + 1)]
    
    # Assigning names to DataFrame columns
    df.columns = interval_names
    
    # Creating a dictionary to store distribution parameters for each interval
    interval_distributions = {}
    interval_pdf_functions = {}
    
    # Iterating through each interval
    for interval in interval_names:
        # Extract the data for the interval across all days
        interval_data = df[interval]
        
        if not np.isfinite(interval_data).all():
            print(f"Warning: {interval} in {filename} contains non-finite values.")
            interval_data = interval_data.dropna()  # drop NaNs
            interval_data = interval_data[np.isfinite(interval_data)]  # drop infinite values
        
        # Fitting a Gaussian distribution to the data
        mu, sigma = norm.fit(interval_data)

        if sigma == 0:
            print(f"Warning: All values in {interval} are the same. Replacing sigma with near-zero value.")
            sigma = 1e-10 
        
        # Storing the distribution parameters
        interval_distributions[interval] = (mu, sigma)

        # Store the PDF function for the interval
        interval_pdf_functions[interval] = lambda x, mu=mu, sigma=sigma: norm.pdf(x, mu, sigma)

        # Visualize the original and fitted data
        x = np.linspace(min(interval_data), max(interval_data), 1000)
        pdf_values = norm.pdf(x, mu, sigma)
        
        if show_plots:
            plt.figure(figsize=(8, 6))
            plt.hist(interval_data, bins=15, density=True, alpha=0.6, color='g')
            plt.plot(x, pdf_values, linewidth=2, color='k')
            plt.title(f'Gaussian Distribution Fit for {interval}')
            plt.xlabel('Data')
            plt.ylabel('Density')
            plt.show()
            
            # Creating Q-Q plot
            plt.figure(figsize=(8, 6))
            probplot(interval_data, dist=norm, sparams=(mu, sigma), plot=plt)
            plt.title(f'Q-Q Plot for {interval} (Gaussian distribution)')
            plt.xlabel('Theoretical quantiles')
            plt.ylabel('Ordered Values')
            plt.grid(True)
            plt.show()

    return interval_distributions, interval_pdf_functions

def generate_simulated_data(distribution, load_distributions, solar_distributions, intervals):
    load_simulation = {}
    solar_simulation = {}

    for interval in intervals:
        load_params = load_distributions[interval]
        solar_params = solar_distributions[interval]
        
        # Generate random data (also known as "simulated data") from PDF functions
        if distribution == 'gaussian':
            load_mu, load_sigma = load_params
            solar_mu, solar_sigma = solar_params
            load_simulation[interval] = np.random.normal(load_mu, load_sigma)
            solar_simulation[interval] = np.random.normal(solar_mu, solar_sigma)
        elif distribution == 'beta':
            load_a, load_b, load_loc, load_scale = load_params
            solar_a, solar_b, solar_loc, solar_scale = solar_params
            load_simulation[interval] = np.random.beta(load_a, load_b, size=1) * load_scale + load_loc
            solar_simulation[interval] = np.random.beta(solar_a, solar_b, size=1) * solar_scale + solar_loc
        elif distribution == 'gamma':
            load_a, load_loc, load_scale = load_params
            solar_a, solar_loc, solar_scale = solar_params
            load_simulation[interval] = np.random.gamma(load_a, load_scale, size=1) + load_loc
            solar_simulation[interval] = np.random.gamma(solar_a, solar_scale, size=1) + solar_loc

    return load_simulation, solar_simulation

def read_data_from_file(filename):
    try:
        df = pd.read_csv(filename, header=None)
        return df
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file {filename} is empty.")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filename}. Error: {e}")
        return None


# Main() function
# Choose either beta, gamma, or gaussian PDF
distribution = 'beta'

# Toggle showing plots, True or False
show_plots = True

# How many intervals (columns) are there in the data? 24 = hourly data, 48 = half-hourly data, 96 = 15-minute data
data_intervals = 48

if distribution == 'gamma':
    load_distributions, load_pdf_functions = fit_gamma_distribution('load_data.csv', show_plots)
    solar_distributions, solar_pdf_functions = fit_gamma_distribution('solar_data.csv', show_plots)

elif distribution == 'gaussian':
    load_distributions, load_pdf_functions = fit_gaussian_distribution('load_data.csv', show_plots)
    solar_distributions, solar_pdf_functions = fit_gaussian_distribution('solar_data.csv', show_plots)

elif distribution == 'beta':
    load_distributions, load_pdf_functions = fit_beta_distribution('load_data.csv', show_plots)
    solar_distributions, solar_pdf_functions = fit_beta_distribution('solar_data.csv', show_plots)

intervals = [f"Interval_{i}" for i in range(1, data_intervals+1)]
load_sim, solar_sim = generate_simulated_data(distribution,load_distributions, solar_distributions, intervals)

load_simulation_array = list(load_sim.values())
solar_simulation_array = list(solar_sim.values())


# Start Monte Carlo Approximate Dynamic Programming optimization here

# See Monte Carlo ADP Pseudocodo.txt to continue