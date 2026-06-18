import numpy as np
import scipy.ndimage as ndimage

# For PDE + SPDE
def make_initial_mono(p):
    """Return (S0, Z0) on a (Ny, Nx) grid."""
    Ny, Nx = p['Ny'], p['Nx']

    people_density = p.get('people_density', 1.0)
    S0 = np.full((Ny, Nx), people_density)

    Y, X = np.ogrid[:Ny, :Nx]
    r2 = (Y - p['outbreak_y']) ** 2 + (X - p['outbreak_x']) ** 2
    blob = np.exp(-r2 / (2.0 * p['outbreak_r'] ** 2))

    Z0 = p['outbreak_f'] * blob
    S0 = np.maximum(S0 - Z0, 0.0)
    return S0, Z0

# For PDE 
def make_initial_experiment_multiple(p):
    """
    Return (S0, Z0) on an (Ny, Nx) grid.
    Spawns 4 random, smooth Gaussian outbreak epicenters.
    """
    Ny, Nx = p['Ny'], p['Nx']
    
    np.random.seed(p.get('seed',42))

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
# For PDE
def make_initial_half_populated(p):
    """
    Return (S0, Z0) on an (Ny, Nx) grid.
    Populates the left half with humans and spawns N random Gaussian outbreaks.
    """
    Ny, Nx = p['Ny'], p['Nx']

    np.random.seed(p.get('seed',42))
    
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

import numpy as np

def make_initial_geography(p):
    """
    Return (S0, Z0) on a (Ny, Nx) grid.
    Generates an irregular population landscape of cities and villages.
    """
    Ny, Nx = p['Ny'], p['Nx']

    np.random.seed(p.get('seed',42))

    S0 = np.zeros((Ny, Nx))

    Y, X = np.ogrid[:Ny, :Nx]
    
    # --- 1. GENERATE HUMAN SETTLEMENTS ---
    num_cities = p.get('num_cities', 3)
    num_villages = p.get('num_villages', 15)
    max_people = p.get('max_people', 200.0)

    def drop_settlements(grid, count, peak_max, radius_min, radius_max):
        """Helper to drop Gaussian population centers onto the grid."""
        for _ in range(count):
            cy, cx = np.random.randint(0, Ny), np.random.randint(0, Nx)
            r = np.random.uniform(radius_min, radius_max)
            peak = np.random.uniform(peak_max * 0.5, peak_max)
            
            # Smoothly diminishing density from the center
            r2 = (Y - cy)**2 + (X - cx)**2
            blob = peak * np.exp(-r2 / (2.0 * r**2))
            grid += blob
        return grid

    # Drop Major Cities (Large radius, high peak population)
    S0 = drop_settlements(S0, num_cities, peak_max=max_people, 
                          radius_min=Nx*0.08, radius_max=Nx*0.15)
    
    # Drop Villages (Small radius, lower peak population)
    S0 = drop_settlements(S0, num_villages, peak_max=max_people*0.3, 
                          radius_min=Nx*0.02, radius_max=Nx*0.05)

    # Cap the maximum density so overlapping cities don't exceed your 200 limit
    S0 = np.clip(S0, 0, max_people)
    
    # Optional: Add a tiny baseline population (e.g., 1 person per cell) 
    # so the wilderness isn't a complete vacuum, allowing zombies to slowly migrate
    S0 = np.maximum(S0, 1.0) 

    # --- 2. GENERATE ZOMBIE OUTBREAK ---
    outbreak_f = p.get('outbreak_f',  1.0)
    outbreak_r = p.get('outbreak_r', 2.0)
    num_outbreaks = p.get('num_outbreaks', 4) # Default to 4 if not specified
    Z0 = np.zeros((Ny, Nx))
    for _ in range(num_outbreaks):
        rand_y = np.random.randint(0, Ny)
        rand_x = np.random.randint(0, Nx)
        
        # Calculate the smooth Gaussian blob
        r2 = (Y - rand_y) ** 2 + (X - rand_x) ** 2
        blob = np.exp(-r2 / (2.0 * outbreak_r ** 2))
        
        # Add this horde to the overall zombie map
        Z0 += outbreak_f * blob
    
    # Ensure zombies consume humans in the exact spot they spawn
    S0 = np.maximum(S0 - Z0, 0.0)
    
    return S0, Z0