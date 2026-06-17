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

    # Określenie maksymalnych wartości dla skali pasków kolorów
    s_sc = max(s.max() for s in sS) or 1
    z_sc = max(z.max() for z in sZ) or 1

    # Nieco szersza figura, aby zrobić miejsce na paski po prawej stronie
    fig, ax = plt.subplots(figsize=(8.5, 7))
    fig.patch.set_facecolor('#0d0d0d')

    rgb0 = make_rgb(sS[0], sZ[0], s_sc, z_sc)
    im = ax.imshow(rgb0, origin='lower')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ttl = ax.set_title(f'{title}  t = {ts[0]:.1f}', color='white', fontweight='bold')

    # --- DODAWANIE PASKÓW KOLORÓW (COLORBARS) ---
    # Narzędzie divider dba o to, żeby paski kolorów były równej wysokości co główny obrazek
    divider = make_axes_locatable(ax)

    # 1. Pasek dla populacji ludzi (S)
    cax_s = divider.append_axes("right", size="4%", pad=0.15)
    sm_s = ScalarMappable(norm=Normalize(vmin=0, vmax=s_sc), cmap=CMAP_S)
    cb_s = fig.colorbar(sm_s, cax=cax_s)
    cb_s.set_label('Susceptible (S)', color='#ff7700', fontweight='bold')
    cb_s.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # 2. Pasek dla populacji zombie (Z)
    cax_z = divider.append_axes("right", size="4%", pad=0.7)
    sm_z = ScalarMappable(norm=Normalize(vmin=0, vmax=z_sc), cmap=CMAP_Z)
    cb_z = fig.colorbar(sm_z, cax=cax_z)
    cb_z.set_label('Zombie (Z)', color='#00dd55', fontweight='bold')
    cb_z.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # Funkcja aktualizująca klatki (teraz odświeża tylko obraz, paski zostają statyczne)
    def _update(f):
        im.set_data(make_rgb(sS[f], sZ[f], s_sc, z_sc))
        ttl.set_text(f'{title}  t = {ts[f]:.1f}')
        return [im]

    anim = FuncAnimation(fig, _update, frames=len(ts),
                         interval=1000 // fps, blit=False)

    # Wymuszamy zmianę rozszerzenia na .mp4, jeśli podano .gif
    fname_mp4 = fname.replace('.gif', '.mp4')
    print(f'  Saving {fname_mp4} ({len(ts)} frames) ...')

    # Używamy writera FFMpeg zamiast Pillow
    anim.save(fname_mp4, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f'  Done: {fname_mp4}')

    return fname_mp4