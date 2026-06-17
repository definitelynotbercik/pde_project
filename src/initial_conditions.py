import numpy as np
import scipy.ndimage as ndimage


def make_initial(p):
    """Return (S0, Z0) on a (Ny, Nx) grid."""
    Ny, Nx = p['Ny'], p['Nx']
    S0 = np.ones((Ny, Nx))

    Y, X = np.ogrid[:Ny, :Nx]
    r2 = (Y - p['outbreak_y']) ** 2 + (X - p['outbreak_x']) ** 2
    blob = np.exp(-r2 / (2.0 * p['outbreak_r'] ** 2))

    Z0 = p['outbreak_f'] * blob
    S0 = np.maximum(S0 - Z0, 0.0)
    return S0, Z0


def make_initial_heterogeneous(p):
    """
    Returns (S0, Z0) with a heterogeneous initial human population density.
    S0 is generated using low-pass filtered random noise to simulate 'cities'.
    """
    Ny, Nx = p['Ny'], p['Nx']

    # 1. Generowanie nierównomiernej mapy ludzi (S0)
    # Tworzymy bazowy szum losowy
    raw_noise = np.random.rand(Ny, Nx)

    # Rozmywamy go mocno, aby stworzyć gładkie "skupiska" (miasta/wioski)
    # Parametr sigma (tutaj 8.0) kontroluje wielkość tych skupisk
    smoothed_density = ndimage.gaussian_filter(raw_noise, sigma=8.0)

    # Normalizujemy, aby zagęszczenie wynosiło od 0.1 do 1.0 (zawsze jest jacyś ludzie)
    min_val, max_val = smoothed_density.min(), smoothed_density.max()
    S0 = 0.1 + 0.9 * (smoothed_density - min_val) / (max_val - min_val)

    # 2. Generowanie ogniska zombie (Z0) - bez zmian
    Y, X = np.ogrid[:Ny, :Nx]
    r2 = (Y - p['outbreak_y']) ** 2 + (X - p['outbreak_x']) ** 2
    blob = np.exp(-r2 / (2.0 * p['outbreak_r'] ** 2))

    # Ognisko infekcji zaraża ułamek LOKALNEJ populacji
    Z0 = p['outbreak_f'] * blob * S0

    # Zmniejszamy populację ludzką o liczbę nowych zombie
    S0 = np.maximum(S0 - Z0, 0.0)

    return S0, Z0