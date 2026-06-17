import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


CMAP_S = LinearSegmentedColormap.from_list(
    'humans', ['#0d0d0d', '#4a1a00', '#cc4400', '#ff8800', '#ffcc44', '#ffffaa'])
CMAP_Z = LinearSegmentedColormap.from_list(
    'zombies', ['#0d0d0d', '#002a1a', '#005533', '#00aa55', '#44ff88', '#ccffdd'])

def make_rgb(S, Z, s_scale, z_scale):
    """
    Composite RGB frame:
      orange channel = humans,
      green  channel = zombies,
      yellow         = infection front.
    """
    rgb = np.zeros((*S.shape, 3))
    sn = np.clip(S / max(s_scale, 1e-10), 0, 1)
    zn = np.clip(Z / max(z_scale, 1e-10), 0, 1)
    # Humans -> warm orange
    rgb[..., 0] += sn * 1.0
    rgb[..., 1] += sn * 0.45
    rgb[..., 2] += sn * 0.05
    # Zombies -> toxic green
    rgb[..., 0] += zn * 0.05
    rgb[..., 1] += zn * 0.85
    rgb[..., 2] += zn * 0.15
    return np.clip(rgb, 0, 1)


def make_gif(res, fname, title='', fps=8):
    """Create and save an RGB-composite GIF with side colorbars."""
    sS, sZ, ts = res['snaps_S'], res['snaps_Z'], res['times']

    # maximum value for the bar
    s_sc = max(s.max() for s in sS) or 1
    z_sc = max(z.max() for z in sZ) or 1

    fig, ax = plt.subplots(figsize=(8.5, 7))
    fig.patch.set_facecolor('#0d0d0d')

    rgb0 = make_rgb(sS[0], sZ[0], s_sc, z_sc)
    im = ax.imshow(rgb0, origin='lower')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ttl = ax.set_title(f'{title}  t = {ts[0]:.1f}', color='white', fontweight='bold')

    divider = make_axes_locatable(ax)

    cax_s = divider.append_axes("right", size="4%", pad=0.15)
    sm_s = ScalarMappable(norm=Normalize(vmin=0, vmax=s_sc), cmap=CMAP_S)
    cb_s = fig.colorbar(sm_s, cax=cax_s)
    cb_s.set_label('Susceptible (S)', color='#ff7700', fontweight='bold')
    cb_s.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    cax_z = divider.append_axes("right", size="4%", pad=0.7)
    sm_z = ScalarMappable(norm=Normalize(vmin=0, vmax=z_sc), cmap=CMAP_Z)
    cb_z = fig.colorbar(sm_z, cax=cax_z)
    cb_z.set_label('Zombie (Z)', color='#00dd55', fontweight='bold')
    cb_z.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    def _update(f):
        im.set_data(make_rgb(sS[f], sZ[f], s_sc, z_sc))
        ttl.set_text(f'{title}  t = {ts[f]:.1f}')
        return [im]

    anim = FuncAnimation(fig, _update, frames=len(ts),
                         interval=1000 // fps, blit=False)

    fname_mp4 = fname.replace('.gif', '.mp4')
    print(f'  Saving {fname_mp4} ({len(ts)} frames) ...')

    anim.save(fname_mp4, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f'  Done: {fname_mp4}')

    return fname_mp4

def day_0(S0,Z0,params):
    fig, axes = plt.subplots(1, 2)

    im1 = axes[0].imshow(S0, origin='lower', cmap=CMAP_S, vmin=0, vmax=params['people_density'])
    axes[0].set_title('Initial S (Healthy Human Density)', color='#ffcc44', fontweight='bold')
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Human Count per Cell')

    im2 = axes[1].imshow(Z0, origin='lower', cmap=CMAP_Z, vmin=0, vmax=params['outbreak_f'])
    axes[1].set_title('Initial Z (Patient Zero Epicenter)', color='#44ff88', fontweight='bold')
    plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Zombie Count per Cell')

    plt.suptitle('Day 0: The Calm Before the Global Storm', fontsize=16, color='white', fontweight='bold')
    plt.tight_layout()
    plt.show()

def model1_population_plot(history_S_total, history_Z_total):
    """Plots the total population dynamics over time using the dark theme palette."""
    # Create a figure with 2 vertically stacked subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # Determine the takeover point if it exists
    takeover_step = None
    if np.any(np.array(history_Z_total) > np.array(history_S_total)):
        takeover_step = np.argmax(np.array(history_Z_total) > np.array(history_S_total))

    # --- Subplot 1: Susceptible Humans ---
    ax1.plot(history_S_total, label='Susceptible Humans (S)', color='#ff8800', linewidth=3)
    if takeover_step is not None:
        ax1.axvline(x=takeover_step, color='white', linestyle='--', alpha=0.5, label='Takeover Point')

    ax1.set_title('Total Population Dynamics Over Time', color='white', fontweight='bold')
    ax1.set_ylabel('Human Count')
    ax1.set_ylim(0, 1.05 * max(history_S_total))  # Gives a nice 5% padding at the top
    ax1.legend()
    ax1.grid(True, color='#333333')

    # --- Subplot 2: Zombies ---
    ax2.plot(history_Z_total, label='Zombies (Z)', color='#00aa55', linewidth=3)
    if takeover_step is not None:
        ax2.axvline(x=takeover_step, color='white', linestyle='--', alpha=0.5, label='Takeover Point')

    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Zombie Count')
    ax2.set_ylim(0, 1.05 * max(history_Z_total))  # Gives a nice 5% padding at the top
    ax2.legend()
    ax2.grid(True, color='#333333')

    # Adjust layout so labels don't overlap
    plt.tight_layout()
    plt.show()