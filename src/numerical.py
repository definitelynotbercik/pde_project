import numpy as np
from scipy.ndimage import laplace


# def laplacian_2d(u, h2_inv):
#     """
#     5-point Laplacian with Neumann BCs (edge-padding).
#     Returns array of same shape as u.
#     """
#     up = np.pad(u, 1, mode='edge')
#     return (up[2:, 1:-1] + up[:-2, 1:-1] +
#             up[1:-1, 2:] + up[1:-1, :-2] -
#             4 * up[1:-1, 1:-1]) * h2_inv

def laplacian_2d(u, h2_inv):
    """
    Computes the 5-point Laplacian using SciPy's optimized C-backend.
    mode='nearest' enforces the zero-flux (Neumann) boundaries.
    """
    # SciPy calculates the raw stencil, and we multiply it by the inverse grid spacing
    return laplace(u, mode='nearest') * h2_inv


def advection_div_upwind(carrier, field, h_inv, is_panic=True):
    """
    Divergence using Upwind scheme to prevent non-physical oscillations.
    is_panic=True  -> humans flee (-grad Z)
    is_panic=False -> zombies hunt (+grad S)
    """
    ny, nx = carrier.shape

    # calculate gradient on the edges
    dFx = (field[:, 1:] - field[:, :-1]) * h_inv
    dFy = (field[1:, :] - field[:-1, :]) * h_inv

    # direction of the flow
    v_sign_x = -dFx if is_panic else dFx
    v_sign_y = -dFy if is_panic else dFy

    # upwind
    Cx = np.where(v_sign_x > 0, carrier[:, :-1], carrier[:, 1:])
    Cy = np.where(v_sign_y > 0, carrier[:-1, :], carrier[1:, :])

    # calculate stream
    Fx = Cx * dFx
    Fy = Cy * dFy

    # divergence according to Neumann conditions
    div = np.zeros((ny, nx))
    div[:, 1:-1] += (Fx[:, 1:] - Fx[:, :-1]) * h_inv
    div[:, 0]  += Fx[:, 0]  * h_inv
    div[:, -1] -= Fx[:, -1] * h_inv

    div[1:-1, :] += (Fy[1:, :] - Fy[:-1, :]) * h_inv
    div[0, :]  += Fy[0, :]  * h_inv
    div[-1, :] -= Fy[-1, :] * h_inv

    return div