#!/usr/bin/env python3
"""
wfl_snap2fig.py -- render one or more snapshot frames side by side with proper x/z axes (in metres) and
a shared colorbar, using matplotlib. Companion to wfl_snap2png.py (which has no axes/colorbar).

  wfl_snap2fig.py out.png nx nz dx dz clip frame file1 [file2 ...] [--labels L1,L2,...] [--diff]
                  [--box x1 x2 z1 z2] [--title T] [--cmap gray|seismic]

--diff: render file1 - file2 as a single panel instead of each file as its own panel.
--labels: comma-separated panel titles (default: file basenames).
--box x1 x2 z1 z2: draw a horizontal green line/box outline at these model coordinates (metres).
"""
import sys, struct, array
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_frame(fn, frame, nx, nz):
    tr = 240 + 4*nz
    with open(fn, 'rb') as f:
        f.seek(frame*nx*tr)
        b = f.read(nx*tr)
    data = np.zeros((nz, nx), dtype=np.float32)
    for ix in range(nx):
        a = array.array('f'); a.frombytes(b[ix*tr+240:(ix+1)*tr])
        data[:, ix] = np.frombuffer(a, dtype=np.float32)
    return data

if __name__ == '__main__':
    a = sys.argv[1:]
    out = a[0]; nx = int(a[1]); nz = int(a[2]); dx = float(a[3]); dz = float(a[4])
    clip = float(a[5]); frame = int(a[6])
    files = []; i = 7
    labels = None; diff = False; box = None; title = None; cmap = 'gray'
    while i < len(a):
        if a[i] == '--labels': labels = a[i+1].split(','); i += 2
        elif a[i] == '--diff': diff = True; i += 1
        elif a[i] == '--box': box = [float(v) for v in a[i+1:i+5]]; i += 5
        elif a[i] == '--title': title = a[i+1]; i += 2
        elif a[i] == '--cmap': cmap = a[i+1]; i += 2
        else: files.append(a[i]); i += 1

    if diff:
        d0 = load_frame(files[0], frame, nx, nz)
        d1 = load_frame(files[1], frame, nx, nz)
        panels = [d0 - d1]
        if labels is None: labels = ['difference']
    else:
        panels = [load_frame(f, frame, nx, nz) for f in files]
        if labels is None: labels = [f.split('/')[-1] for f in files]

    extent = [0, nx*dx, nz*dz, 0]  # x: 0..nx*dx ; z: nz*dz..0 (depth increases downward)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2*n, 5.2), squeeze=False, constrained_layout=True)
    axes = axes[0]
    im = None
    for ax, panel, lab in zip(axes, panels, labels):
        im = ax.imshow(panel, cmap=cmap, vmin=-clip, vmax=clip, extent=extent, aspect='auto')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(lab)
        if box is not None:
            x1, x2, z1, z2 = box
            ax.plot([x1, x2], [z1, z1], color='limegreen', linewidth=1.2)
            if z2 != z1:
                ax.plot([x1, x2], [z2, z2], color='limegreen', linewidth=1.2)
                ax.plot([x1, x1], [z1, z2], color='limegreen', linewidth=1.2)
                ax.plot([x2, x2], [z1, z2], color='limegreen', linewidth=1.2)
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label('amplitude')
    if title: fig.suptitle(title)
    fig.savefig(out, dpi=130)
    print('wrote', out)
