import numpy as np
import time
from src.numerical import *
from scipy.ndimage import gaussian_filter

# SPDE
def solve_model2(S0, Z0, p, seed=42, tag='M2'):
    """
    Model 2: stochastic PDE (Euler-Maruyama)
    """
    Ds, Dz = p['D_s'], p['D_z']
    beta = p['beta']
    alpha = p.get('alpha_m2', p['alpha'])  # M2 uses lower alpha (natural decay)
    kp, kh = p['k_panic'], p['k_hunt']
    sigma = p['sigma']
    dt, h = p['dt'], p['h']
    N = p['n_steps']
    si = p['snap_every']
    h_inv = 1.0 / h
    h2_inv = 1.0 / h ** 2
    sqrt_dt = np.sqrt(dt)

    rng = np.random.default_rng(seed)
    S, Z = S0.copy(), Z0.copy()
    snaps_S, snaps_Z, ts = [S.copy()], [Z.copy()], [0.0]
    hist_S, hist_Z = [S.sum()], [Z.sum()]
    t0 = time.time()

    for n in range(1, N + 1):
        # calculate spatial operators
        lap_S = laplacian_2d(S, h2_inv)
        lap_Z = laplacian_2d(Z, h2_inv)

        # upwind
        div_panic = advection_div_upwind(S, Z, h_inv, is_panic=True)  # div(S grad Z)
        div_hunt = advection_div_upwind(Z, S, h_inv, is_panic=False)  # div(Z grad S)

        SZ = S * Z

        # wiener increment
        noise = sigma * S * rng.standard_normal(S.shape) * sqrt_dt

        # time step (Euler-Maruyama)
        S_new = S + dt * (Ds * lap_S + kp * div_panic - beta * SZ) + noise
        Z_new = Z + dt * (Dz * lap_Z - kh * div_hunt + beta * SZ - alpha * Z)


        # --- HUMANS (S) ---
        S_neg_mask = S_new < 0
        added_mass_S = np.abs(S_new[S_neg_mask]).sum()

        S = np.maximum(S_new, 0.0)

        S_pos_mask = S > 0
        if added_mass_S > 0 and np.any(S_pos_mask):
            S[S_pos_mask] -= (added_mass_S / np.sum(S_pos_mask))
            S = np.maximum(S, 0.0)

        # --- ZOMBIES (Z) ---
        Z_neg_mask = Z_new < 0
        added_mass_Z = np.abs(Z_new[Z_neg_mask]).sum()

        Z = np.maximum(Z_new, 0.0)

        Z_pos_mask = Z > 0
        if added_mass_Z > 0 and np.any(Z_pos_mask):
            Z[Z_pos_mask] -= (added_mass_Z / np.sum(Z_pos_mask))
            Z = np.maximum(Z, 0.0)

        # save history and plots
        hist_S.append(S.sum())
        hist_Z.append(Z.sum())
        if n % si == 0:
            snaps_S.append(S.copy())
            snaps_Z.append(Z.copy())
            ts.append(n * dt)
        if n % max(N // 5, 1) == 0:
            print(f'  [{tag}] step {n}/{N}  t={n * dt:.1f}'
                  f'  S={S.sum():.1f}  Z={Z.sum():.1f}'
                  f'  Zmax={Z.max():.4f}  ({time.time() - t0:.1f}s)')

    print(f'  [{tag}] done in {time.time() - t0:.1f}s')
    return dict(snaps_S=snaps_S, snaps_Z=snaps_Z, times=ts,
                hist_S=np.array(hist_S), hist_Z=np.array(hist_Z),
                S_final=S, Z_final=Z, dt=dt)

def model1_base_system(S, Z, dx, dy, dt, Ds, Dz, beta, alpha):
    # Only calculate the inverse grid spacing needed for diffusion
    h2_inv = 1.0 / (dx * dy) 

    # 1. Pure Diffusion
    laplacian_S = laplacian_2d(S, h2_inv)
    laplacian_Z = laplacian_2d(Z, h2_inv)
    
    # Notice: div_panic has been entirely removed from the base model.
    # The base model should only handle diffusion and infection/decay.

    # 2. Forward Euler Time Integration
    S_new = S + dt * (Ds * laplacian_S - beta * S * Z)
    Z_new = Z + dt * (Dz * laplacian_Z + beta * S * Z - alpha * S * Z)

    # 3. Enforce Non-negativity
    return np.maximum(S_new, 0), np.maximum(Z_new, 0)

def model1_pure_system(S, Z, dx, dy, dt, Ds, Dz):
    """
    Pure Diffusion Model:
    dS/dt = Ds * Laplace(S)
    dZ/dt = Dz * Laplace(Z)
    """
    # Calculate inverse step sizes based on grid spacing
    h2_inv = 1.0 / (dx * dy)
    
    # 1. Spatial Diffusion (Custom Laplacian Approximation)
    laplacian_S = laplacian_2d(S, h2_inv)
    laplacian_Z = laplacian_2d(Z, h2_inv)

    # 2. Final Explicit Update Equations (Forward Euler)
    # The advection (panic) and reaction (infection) terms have been entirely removed.
    S_new = S + dt * (Ds * laplacian_S)
    Z_new = Z + dt * (Dz * laplacian_Z)

    # Return the updated grids
    return np.maximum(S_new, 0), np.maximum(Z_new, 0)

def model1_updated_system(S, Z, dx, dy, dt, Ds, Dz, beta, alpha, k_panic, fear_radius):
    # Calculate inverse step sizes (Assuming dx == dy for simplicity, otherwise use dx**2 and dy**2 separately inside the laplacian function)
    h_inv = 1.0 / dx
    h2_inv = 1.0 / (dx**2) 

    # 1. Spatial Diffusion
    laplacian_S = laplacian_2d(S, h2_inv)
    laplacian_Z = laplacian_2d(Z, h2_inv)

    # 2. Fear Field
    Z_fear = gaussian_filter(Z, sigma=fear_radius)

    # 3. Divergence of the panic term (Advection)
    div_panic = advection_div_upwind(S, Z_fear, h_inv)

    # 4. Reaction Terms (Rates)
    actual_bites = beta * S * Z
    zombie_deaths = alpha * S * Z

    # 5. Explicit Update Equations (dt applied to ALL terms!)
    S_new = S + dt * (Ds * laplacian_S + k_panic * div_panic - actual_bites)
    Z_new = Z + dt * (Dz * laplacian_Z - zombie_deaths + actual_bites)

    return np.maximum(S_new, 0), np.maximum(Z_new, 0)

def model1_simulation_loop(sim_type, S, Z, p):
    dx, dy, dt = p['dx'], p['dy'], p['dt']
    Ds, Dz, beta, alpha = p['Ds'], p['Dz'], p['beta'], p['alpha']
    k_panic, fear_radius = p['k_panic'], p['fear_radius']
    NUM_STEPS = p['NUM_STEPS']
    
    # Check if snap_every is in params, otherwise default to 10
    snap_every = p.get('plot_interval', 10)

    # CFL STABILITY CHECK
    D_max = max(Ds, Dz)
    dt_cfl = (dx * dy) / (4.0 * D_max)
    if dt > dt_cfl:
        dt = dt_cfl * 0.9 
        print(f"⚠️ WARNING: dt={p['dt']} is unstable! Reducing to safe limit: dt={dt:.5f}")
    
    # 1D arrays for the line graphs
    history_S_total, history_Z_total = [], []

    # 3D arrays (lists of 2D grids) for the GIF
    history_S_grids, history_Z_grids = [], []

    if sim_type=='base':
        for step in range(NUM_STEPS):
            S, Z = model1_base_system(S, Z, dx, dy, dt, Ds, Dz, beta, alpha)
        
            history_S_total.append(np.sum(S))
            history_Z_total.append(np.sum(Z))

            if step % snap_every == 0:
                history_S_grids.append(S.copy()) # .copy() is strictly required here
                history_Z_grids.append(Z.copy())

    elif sim_type=='updated':
        for step in range(NUM_STEPS):
            S, Z = model1_updated_system(S, Z, dx, dy, dt, Ds, Dz, beta, alpha, k_panic, fear_radius)
        
            history_S_total.append(np.sum(S))
            history_Z_total.append(np.sum(Z))

            if step % snap_every == 0:
                history_S_grids.append(S.copy()) # .copy() is strictly required here
                history_Z_grids.append(Z.copy())
    elif sim_type=='pure':
        for step in range(NUM_STEPS):
            S, Z = model1_pure_system(S, Z, dx, dy, dt, Ds, Dz)
        
            history_S_total.append(np.sum(S))
            history_Z_total.append(np.sum(Z))

            if step % snap_every == 0:
                history_S_grids.append(S.copy()) # .copy() is strictly required here
                history_Z_grids.append(Z.copy())
            
    return history_S_total, history_Z_total, history_S_grids, history_Z_grids