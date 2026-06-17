import numpy as np
import scipy.ndimage as ndimage


def make_initial_mono(p):
    """Return (S0, Z0) on a (Ny, Nx) grid."""
    Ny, Nx = p['Ny'], p['Nx']
    S0 = np.ones((Ny, Nx))

    Y, X = np.ogrid[:Ny, :Nx]
    r2 = (Y - p['outbreak_y']) ** 2 + (X - p['outbreak_x']) ** 2
    blob = np.exp(-r2 / (2.0 * p['outbreak_r'] ** 2))

    Z0 = p['outbreak_f'] * blob
    S0 = np.maximum(S0 - Z0, 0.0)
    return S0, Z0


def make_initial_hetero(p, seed=42):
    """
    Returns (S0, Z0) with a heterogeneous initial human population density.
    S0 is generated using low-pass filtered random noise to simulate 'cities'.
    """
    Ny, Nx = p['Ny'], p['Nx']

    rng = np.random.default_rng(seed)

    # create noise
    raw_noise = np.random.rand(Ny, Nx)

    # smoothen the noise
    smoothed_density = ndimage.gaussian_filter(raw_noise, sigma=8.0)

    # normalize
    min_val, max_val = smoothed_density.min(), smoothed_density.max()
    S0 = 0.1 + 0.9 * (smoothed_density - min_val) / (max_val - min_val)

    # generate zombie center
    Y, X = np.ogrid[:Ny, :Nx]
    r2 = (Y - p['outbreak_y']) ** 2 + (X - p['outbreak_x']) ** 2
    blob = np.exp(-r2 / (2.0 * p['outbreak_r'] ** 2))

    # zombies in the center infect some of the humans
    Z0 = p['outbreak_f'] * blob * S0

    # substract this number of zombies from human population
    S0 = np.maximum(S0 - Z0, 0.0)

    return S0, Z0