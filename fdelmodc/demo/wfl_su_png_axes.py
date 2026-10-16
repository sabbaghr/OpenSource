#!/usr/bin/env python3
"""
wfl_su_png_axes.py -- render an SU gather (one trace per x-position, nz samples per trace) as a
grayscale (diverging, zero=mid-grey) PNG with labeled x/z axes in meters, using matplotlib.

  wfl_su_png_axes.py su_file nx nz dx dz out_png [title]
"""
import sys, struct, array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read_su(fn, nx, nz):
    tr = 240 + 4*nz
    img = np.zeros((nx, nz), dtype=np.float32)
    with open(fn, 'rb') as f:
        for ix in range(nx):
            b = f.read(tr)
            a = array.array('f'); a.frombytes(b[240:])
            img[ix, :] = a
    return img

if __name__ == '__main__':
    su_file, nx, nz, dx, dz, out_png = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
    title = sys.argv[7] if len(sys.argv) > 7 else None

    img = read_su(su_file, nx, nz)
    clip = np.percentile(np.abs(img), 98)
    if clip == 0: clip = 1.0

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    extent = [0, (nx-1)*dx, (nz-1)*dz, 0]
    ax.imshow(img.T, cmap='gray', vmin=-clip, vmax=clip, extent=extent, aspect='auto')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    if title: ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print('wrote', out_png, 'clip=%.3e' % clip)
