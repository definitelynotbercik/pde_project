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


def make_gif(res, fname, title='', fps=8, params=None, cap_human=False):
    """Create and save an RGB-composite MP4 with dynamically scaling side colorbars."""
    sS, sZ, ts = res['snaps_S'], res['snaps_Z'], res['times']

    # --- Inicjalizacja dla klatki 0 ---
    initial_max_s = np.max(sS[0]) if np.max(sS[0]) > 0 else 1.0
    if cap_human and params and 'people_density' in params:
        initial_max_s = params['people_density']

    initial_max_z = np.max(sZ[0]) if np.max(sZ[0]) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8.5, 7))
    fig.patch.set_facecolor('#0d0d0d')

    # Przekazujemy początkowe maksima do funkcji make_rgb
    rgb0 = make_rgb(sS[0], sZ[0], initial_max_s, initial_max_z)
    im = ax.imshow(rgb0, origin='lower')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ttl = ax.set_title(f'{title}  t = {ts[0]:.1f}', color='white', fontweight='bold')

    divider = make_axes_locatable(ax)

    # Inicjalizacja pasków kolorów (początkowa skala)
    cax_s = divider.append_axes("right", size="4%", pad=0.15)
    sm_s = ScalarMappable(norm=Normalize(vmin=0, vmax=initial_max_s), cmap=CMAP_S)
    cb_s = fig.colorbar(sm_s, cax=cax_s)
    cb_s.set_label('Susceptible (S)', color='#ff7700', fontweight='bold')
    cb_s.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    cax_z = divider.append_axes("right", size="4%", pad=0.9)  # pad=0.9 by uniknąć nachodzenia tekstu
    sm_z = ScalarMappable(norm=Normalize(vmin=0, vmax=initial_max_z), cmap=CMAP_Z)
    cb_z = fig.colorbar(sm_z, cax=cax_z)
    cb_z.set_label('Zombie (Z)', color='#00dd55', fontweight='bold')
    cb_z.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # --- Pętla aktualizująca klatki ---
    def _update(f):
        current_S = sS[f]
        current_Z = sZ[f]

        # 1. DYNAMICZNE SKALOWANIE: Znalezienie maksimów w obecnej klatce
        current_max_s = np.max(current_S) if np.max(current_S) > 0 else 1.0
        if cap_human and params and 'people_density' in params:
            current_max_s = params['people_density']

        current_max_z = np.max(current_Z) if np.max(current_Z) > 0 else 1.0

        # 2. Aktualizacja obrazu RGB z relatywnym skalowaniem
        im.set_data(make_rgb(current_S, current_Z, current_max_s, current_max_z))
        ttl.set_text(f'{title}  t = {ts[f]:.1f}')

        # 3. Przerysowanie pasków legendy z nowymi limitami (set_clim)
        sm_s.set_clim(vmin=0, vmax=current_max_s)
        cb_s.update_normal(sm_s)  # Wymusza odświeżenie etykiet osi paska S

        sm_z.set_clim(vmin=0, vmax=current_max_z)
        cb_z.update_normal(sm_z)  # Wymusza odświeżenie etykiet osi paska Z

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

def population_dynamics(res, filename):

    t_arr2 = np.arange(len(res['hist_S'])) * res['dt']

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_arr2, res['hist_S'], color='#ff8800', lw=2, label='S (susceptible)')
    ax.plot(t_arr2, res['hist_Z'], color='#00cc55', lw=2, label='Z (zombie)')
    ax.plot(t_arr2, res['hist_S'] + res['hist_Z'],
            color='#aaaaaa', lw=1, ls='--', label='S + Z')
    ax.set_xlabel('Time');
    ax.set_ylabel('Total density')
    ax.set_title('Model 2 — Population Dynamics',
                 fontweight='bold', color='#ffcc66')
    ax.legend(fontsize=12, framealpha=0.3);
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=140, bbox_inches='tight')
    plt.show()

def model1_generate_gif(grid_history_S, grid_history_Z, filename,params, cap_human=True):
    """
    Generates an animated GIF of the spatial dynamics.
    Humans use static scaling (to watch the lights go out).
    Zombies use dynamic scaling (to track the horde's shape at all times).
    """
    
    # Create figure with the dark apocalyptic theme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 1. Initialize Human Subplot (Static Scaling) ---
    cax1 = ax1.imshow(grid_history_S[0], cmap=CMAP_S, origin='lower', vmin=0)
    ax1.set_title('Human Density (S)', color='#ffcc44', fontweight='bold')
    fig.colorbar(cax1, ax=ax1, shrink=0.8, label='Absolute Human Count')

    # --- 2. Initialize Zombie Subplot (Dynamic Scaling) ---
    initial_max_z = np.max(grid_history_Z[0]) if np.max(grid_history_Z[0]) > 0 else 1.0
    cax2 = ax2.imshow(grid_history_Z[0], cmap=CMAP_Z, origin='lower', vmin=0, vmax=initial_max_z)
    ax2.set_title('Zombie Density (Z)', color='#44ff88', fontweight='bold')
    fig.colorbar(cax2, ax=ax2, shrink=0.8, label='Relative Horde Density')

    fig.suptitle('Day 0.0: The Calm Before the Storm', fontsize=16, color='white', fontweight='bold')
    plt.tight_layout()

    # --- 3. Animation Update Loop ---
    def update(frame):
        current_S = grid_history_S[frame]
        current_Z = grid_history_Z[frame]
        
        # Update the visual grid matrices
        cax1.set_data(current_S)
        cax2.set_data(current_Z)
        
        # DYNAMIC SCALING FOR ZOMBIES: 
        # Find the highest concentration of zombies in this specific frame
        current_max_S = np.max(current_S) if np.max(current_S) > 0 else 1
        current_max_Z = np.max(current_Z) if np.max(current_Z) > 0 else 1
        
        # Apply the new maximums to the heatmaps so the colorbar rescales
        cax1.set_clim(vmin=0, vmax=params['people_density'] if cap_human else current_max_S)
        cax2.set_clim(vmin=0, vmax=current_max_Z)
        
        # Update Title (Translating frames * interval into "Days")
        current_day = frame*params['plot_interval']
        fig.suptitle(f'Global Outbreak Progress - Day {current_day:.1f}', fontsize=16, color='white', fontweight='bold')
        
        return cax1, cax2

    ani = FuncAnimation(fig, update, frames=len(grid_history_S), blit=False)
    
    # Close the plot to prevent it from displaying a duplicate static image
    ani.save(f"plots/{filename}", writer=PillowWriter(fps=15))
    plt.close(fig)
    
    # Render and return the animation inline
    return ani