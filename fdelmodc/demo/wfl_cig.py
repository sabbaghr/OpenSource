#!/usr/bin/env python3
"""
wfl_cig.py -- shot-index common image gathers from per-shot RTM images.

  wfl_cig.py root image_rel dx dz "xs1 xs2 ..." "x01 x02 ..." [--kinds full,atten] [--zclip 300] [--interp 50]

Reads <root>/shot_x<xs>/<image_rel>_<kind>.su (one image per shot, nx traces of nz samples, trace ix
at x = ix*dx), and for each CIG location x0 takes column ix0 = round(x0/dx) of every shot image:
    CIG_x0(z, xs) = I_xs(x0, z)
Writes <root>/cig_x<x0>_<kind>.su (one trace per shot, offset header = xs, ascending), the stacked image
<root>/stack_<kind>.su (sum over shots), and PNGs:
    <root>/cig_<kind>.png    all CIGs side by side (density only); for DISPLAY the gather is linearly
                             interpolated along the source-x axis onto a --interp m grid (0 = off) --
                             cosmetic only, the .su gathers keep the actual shots
    <root>/shot_x<xs>.png    each shot's own image (full and atten), common clip across shots
    <root>/stack.png         stacked images, one panel per kind, CIG locations marked
Clip = 99th percentile of |amplitude| below --zclip metres (excludes the near-surface source/line
artifacts, which otherwise set the colour scale).
"""
import sys, os, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

REFL_Z = 1500.0

def read_su(fn):
    d = open(fn, 'rb').read()
    ns = struct.unpack('<H', d[114:116])[0]; tr = 240 + 4*ns
    n = len(d)//tr
    return np.array([np.frombuffer(d[i*tr+240:(i+1)*tr], '<f4') for i in range(n)])   # (ntr, ns)

def write_su(fn, traces, dz, offsets=None):
    out = bytearray()
    for i, t in enumerate(traces):
        h = bytearray(240)
        struct.pack_into('<i', h, 0, i+1); struct.pack_into('<i', h, 4, i+1)
        if offsets is not None: struct.pack_into('<i', h, 36, int(offsets[i]))
        struct.pack_into('<H', h, 114, len(t)); struct.pack_into('<H', h, 116, 1)
        struct.pack_into('<f', h, 180, float(dz))              # d1
        out += h; out += np.asarray(t, '<f4').tobytes()
    open(fn, 'wb').write(out)

def clipval(a, iz0):
    v = np.abs(a[..., iz0:]).ravel()
    return np.percentile(v, 99) if v.size and v.max() > 0 else 1.0

if __name__ == '__main__':
    root, rel = sys.argv[1], sys.argv[2]
    dx, dz = float(sys.argv[3]), float(sys.argv[4])
    shots = [int(float(s)) for s in sys.argv[5].split()]
    x0s = [int(float(s)) for s in sys.argv[6].split()]
    rest = sys.argv[7:]
    kinds = rest[rest.index('--kinds')+1].split(',') if '--kinds' in rest else ['full', 'atten']
    zclip = float(rest[rest.index('--zclip')+1]) if '--zclip' in rest else 300.0
    dinterp = float(rest[rest.index('--interp')+1]) if '--interp' in rest else 50.0
    shots = sorted(shots)
    iz0 = int(round(zclip/dz))
    stacks = {}
    allimgs = {}

    for kind in kinds:
        imgs = np.array([read_su(os.path.join(root, 'shot_x%d' % xs, '%s_%s.su' % (rel, kind))) for xs in shots])
        nsh, nx, nz = imgs.shape
        z = np.arange(nz)*dz
        stack = imgs.sum(axis=0)
        write_su(os.path.join(root, 'stack_%s.su' % kind), stack, dz)
        gathers = {}
        for x0 in x0s:
            ix0 = int(round(x0/dx))
            g = imgs[:, ix0, :]                                 # (nshot, nz)
            gathers[x0] = g
            write_su(os.path.join(root, 'cig_x%d_%s.su' % (x0, kind)), g, dz, offsets=shots)
            iz_pk = iz0 + int(np.argmax(np.abs(g[:, iz0:]).sum(axis=0)))
            print('%-5s CIG x0=%5d: peak |I| (summed over shots) at z=%.0f m (reflector %.0f);'
                  ' per-shot peak depth: %s' % (kind, x0, iz_pk*dz, REFL_Z,
                  ' '.join('%d:%.0f' % (xs, (iz0+np.argmax(np.abs(t[iz0:])))*dz) for xs, t in zip(shots, g))))

        # --- all CIGs for this kind, one panel each, shared clip
        c = clipval(np.array(list(gathers.values())), iz0)
        fig, axs = plt.subplots(1, len(x0s), figsize=(3.2*len(x0s), 6), sharey=True, squeeze=False)
        for ax, x0 in zip(axs[0], x0s):
            g = gathers[x0]
            if dinterp > 0 and nsh > 1:
                xi = np.arange(shots[0], shots[-1]+0.5*dinterp, dinterp)
                g = np.array([np.interp(xi, shots, g[:, iz]) for iz in range(nz)]).T
            ext = [shots[0], shots[-1], z[-1], z[0]]
            ax.imshow(g.T, aspect='auto', cmap='gray_r', vmin=-c, vmax=c, extent=ext, interpolation='bilinear')
            ax.set_title('CIG x0=%d m' % x0); ax.set_xlabel('source x (m)')
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5)); ax.tick_params(axis='x', labelrotation=45)
        axs[0][0].set_ylabel('depth (m)')
        fig.suptitle('Shot-index CIGs, energy-norm image (%s), smooth migration model' % kind)
        fig.tight_layout()
        fn = os.path.join(root, 'cig_%s.png' % kind); fig.savefig(fn, dpi=130); plt.close(fig)
        print('wrote', fn)
        stacks[kind] = stack
        allimgs[kind] = imgs

    # --- individual shot images, common clip per kind across all shots
    clips = {k: clipval(v, iz0) for k, v in allimgs.items()}
    for ish, xs in enumerate(shots):
        fig, axs = plt.subplots(len(allimgs), 1, figsize=(11, 3.4*len(allimgs)), squeeze=False)
        for ax, (kind, imgs) in zip(axs[:, 0], allimgs.items()):
            im = imgs[ish]; nx, nz = im.shape; c = clips[kind]
            ax.imshow(im.T, aspect='auto', cmap='gray_r', vmin=-c, vmax=c, extent=[0, (nx-1)*dx, (nz-1)*dz, 0])
            ax.plot([xs], [30], 'r*', ms=11, clip_on=False)
            ax.set_title('shot x=%d m (%s)' % (xs, kind)); ax.set_ylabel('depth (m)')
        axs[-1, 0].set_xlabel('x (m)')
        fig.tight_layout()
        fn = os.path.join(root, 'shot_x%d.png' % xs); fig.savefig(fn, dpi=110); plt.close(fig)
        print('wrote', fn)

    # --- stacked images
    st = stacks
    fig, axs = plt.subplots(len(st), 1, figsize=(11, 3.4*len(st)), squeeze=False)
    for ax, (kind, s) in zip(axs[:, 0], st.items()):
        nx, nz = s.shape
        c = clipval(s, iz0)
        ax.imshow(s.T, aspect='auto', cmap='gray_r', vmin=-c, vmax=c, extent=[0, (nx-1)*dx, (nz-1)*dz, 0])
        for x0 in x0s: ax.axvline(x0, color='c', lw=0.8)
        ax.plot(shots, [30]*len(shots), 'r*', ms=9, clip_on=False)
        ax.set_title('stack of %d shots (%s)' % (len(shots), kind)); ax.set_ylabel('depth (m)')
    axs[-1, 0].set_xlabel('x (m)')
    fig.tight_layout()
    fn = os.path.join(root, 'stack.png'); fig.savefig(fn, dpi=110); plt.close(fig)
    print('wrote', fn)
