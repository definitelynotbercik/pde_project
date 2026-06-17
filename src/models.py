import numpy as np
import time
from src.numerical import *

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