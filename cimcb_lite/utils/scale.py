import numpy as np


def scale(x, axis=0, ddof=1, method='auto', return_mu_sigma=False, mu='default', sigma='default'):
    """Scales x (which can include nans) with method: 'auto', 'pareto', 'vast', 'range', or 'level'."""
    
    x = np.asanyarray(x)
    
    # Calculate mu and sigma if set to 'default'
    if mu is 'default':
        mu = np.nanmean(x, axis=axis)
    if sigma is 'default':
        sigma = np.nanstd(x, axis=axis, ddof=ddof)
        sigma = np.where(sigma > 0, sigma, 1) # if a value in sigma equals 0 it is converted to 1
    
    # expands the shape of the array when required
    if axis and mu.ndim < x.ndim:
        mu = np.expand_dims(mu, axis=axis)
        sigma = np.expand_dims(sigma, axis=axis)
        
    # Error check before scaling
    if len(mu) != len(x.T):
        raise ValueError("Length of mu array does not match x matrix.")
    if len(sigma) != len(x.T):
        raise ValueError("Length of sigma array does not match x matrix.")

    # Scale based on selected method
    if method is "auto":
        z = (x - mu) / sigma
    elif method is "pareto":
        z = (x - mu) / np.sqrt(sigma)
    elif method is "vast":
        z = ((x - mu) / sigma) * (mu / sigma)
    elif method is "level":
        z = (x - mu) / mu
    else:
        raise ValueError("Method has to be either 'auto', 'pareto', 'vast', or 'level'.")
    
    if return_mu_sigma is True: 
        return mu, sigma
    else:
        return z
