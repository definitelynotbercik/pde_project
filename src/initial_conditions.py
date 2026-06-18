import numpy as np
import scipy.ndimage as ndimage

# For PDE + SPDE
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

# For PDE 
def make_initial_experiment_multiple(p,seed=42):
    """
    Return (S0, Z0) on an (Ny, Nx) grid.
    Spawns 4 random, smooth Gaussian outbreak epicenters.
    """
    Ny, Nx = p['Ny'], p['Nx']
    
    rng = np.random.default_rng(seed)

    # Initialize baseline human population
    people_density = p.get('people_density', 1.0)
    S0 = np.full((Ny, Nx), people_density)
    
    # Initialize empty zombie grid
    Z0 = np.zeros((Ny, Nx))

    # Create coordinate grid for distance calculations
    Y, X = np.ogrid[:Ny, :Nx]
    
    outbreak_f = p.get('outbreak_f',  1.0)
    outbreak_r = p.get('outbreak_r', 2.0)
    num_outbreaks = p.get('num_outbreaks', 4) # Default to 4 if not specified
    # Spawn 4 random outbreaks
    for _ in range(num_outbreaks):
        rand_y = np.random.randint(0, Ny)
        rand_x = np.random.randint(0, Nx)
        
        # Calculate the smooth Gaussian blob
        r2 = (Y - rand_y) ** 2 + (X - rand_x) ** 2
        blob = np.exp(-r2 / (2.0 * outbreak_r ** 2))
        
        # Add this horde to the overall zombie map
        Z0 += outbreak_f * blob
    
    # Ensure humans are depleted where zombies spawn
    S0 = np.maximum(S0 - Z0, 0.0)
    
    return S0, Z0


def make_initial_experiment_multiple_hetero(p, seed=42):
    """
    Return (S0, Z0) on an (Ny, Nx) grid.
    Spawns multiple random, smooth Gaussian outbreak epicenters
    on a heterogeneous human population density map.
    Allows control over city dispersion/isolation via params.
    """
    Ny, Nx = p['Ny'], p['Nx']

    rng = np.random.default_rng(seed)

    city_size = p.get('city_size', 8.0)
    city_isolation = p.get('city_isolation', 1.0)

    raw_noise = rng.random((Ny, Nx))
    smoothed_density = ndimage.gaussian_filter(raw_noise, sigma=city_size)

    min_val, max_val = smoothed_density.min(), smoothed_density.max()
    normalized_density = (smoothed_density - min_val) / (max_val - min_val)

    S0 = 0.1 + 0.9 * (normalized_density ** city_isolation)

    Z0 = np.zeros((Ny, Nx))

    Y, X = np.ogrid[:Ny, :Nx]

    outbreak_f = p.get('outbreak_f', 1.0)
    outbreak_r = p.get('outbreak_r', 2.0)
    num_outbreaks = p.get('num_outbreaks', 4)

    for _ in range(num_outbreaks):
        rand_y = rng.integers(0, Ny)
        rand_x = rng.integers(0, Nx)

        r2 = (Y - rand_y) ** 2 + (X - rand_x) ** 2
        blob = np.exp(-r2 / (2.0 * outbreak_r ** 2))

        Z0 += outbreak_f * blob * S0

    Z0 = np.minimum(Z0, S0)

    S0 = np.maximum(S0 - Z0, 0.0)

    return S0, Z0
# For PDE
def make_initial_half_populated(p,seed=42):
    """
    Return (S0, Z0) on an (Ny, Nx) grid.
    Populates the left half with humans and spawns N random Gaussian outbreaks.
    """
    Ny, Nx = p['Ny'], p['Nx']

    rng = np.random.default_rng(seed)
    
    # 1. Initialize Humans: Fill only the left half
    S0 = np.zeros((Ny, Nx))
    S0[:, :Nx // 2] = p['people_density']
    
    # 2. Initialize Zombies: Empty grid
    Z0 = np.zeros((Ny, Nx))

    # Coordinate grid for Gaussian calculation
    Y, X = np.ogrid[:Ny, :Nx]
    
    outbreak_f = p['outbreak_f']
    outbreak_r = p['outbreak_r']
    num_outbreaks = p.get('num_outbreaks', 4) # Default to 4 if not specified

    # 3. Spawn N random outbreaks
    for _ in range(num_outbreaks):
        rand_y = np.random.randint(0, Ny)
        rand_x = np.random.randint(0, Nx)
        
        # Calculate Gaussian blob
        r2 = (Y - rand_y) ** 2 + (X - rand_x) ** 2
        blob = np.exp(-r2 / (2.0 * outbreak_r ** 2))
        
        Z0 += outbreak_f * blob
    
    # 4. Maintain mass balance: remove humans where zombies are placed
    S0 = np.maximum(S0 - Z0, 0.0)
    
    return S0, Z0


def make_initial_half_populated_m2(p, seed=42):
    """
    Return (S0, Z0) on an (Ny, Nx) grid.
    Populates the left half with humans and spawns N random Gaussian outbreaks.
    """
    Ny, Nx = p['Ny'], p['Nx']

    # Używamy lokalnego generatora dla zachowania pełnej powtarzalności układu
    rng = np.random.default_rng(seed)

    # 1. Initialize Humans: Fill only the left half
    S0 = np.zeros((Ny, Nx))
    S0[:, :Nx // 2] = p.get('people_density', 1.0)

    # 2. Initialize Zombies: Empty grid
    Z0 = np.zeros((Ny, Nx))

    # Coordinate grid for Gaussian calculation
    Y, X = np.ogrid[:Ny, :Nx]

    outbreak_f = p['outbreak_f']
    outbreak_r = p['outbreak_r']
    num_outbreaks = p.get('num_outbreaks', 4)

    # 3. Spawn N random outbreaks
    for _ in range(num_outbreaks):
        rand_y = rng.integers(0, Ny)
        rand_x = rng.integers(0, Nx)

        # Calculate Gaussian blob
        r2 = (Y - rand_y) ** 2 + (X - rand_x) ** 2
        blob = np.exp(-r2 / (2.0 * outbreak_r ** 2))

        # POPRAWKA: Usunięto mnożenie przez S0.
        # Zombie spawnują się niezależnie od tego, czy na danym polu są ludzie.
        Z0 += outbreak_f * blob

    # 4. Maintain mass balance: remove humans where outbreaks overlap with the populated half
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