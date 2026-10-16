#!/usr/bin/env python3
"""
wfl_cig_report.py -- beamer PDF of a multi-shot CIG run (output of wfl_cig_rocha.sh + wfl_cig.py).

  wfl_cig_report.py root outdir "xs1 xs2 ..." "x01 x02 ..." [--title T] [--subtitle S] [--show "xa xb ..."]
                    [--tex name]

Copies cig_{atten,full}.png and stack.png from root into outdir (so later runs that overwrite root/ don't
break the report), draws shots_{atten,full}.png = the --show shots (default: 5 evenly spread) stacked in one
figure per image type, writes outdir/<name>.tex (default: outdir's basename): setup, representative shots
(one slide per image type), CIGs (atten, full), stack, then outdir/observations.tex if it exists (a list of
\\item lines), and compiles it.
"""
import sys, os, shutil, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wfl_cig import read_su, clipval

root, outdir = sys.argv[1], sys.argv[2]
shots = [int(float(s)) for s in sys.argv[3].split()]
x0s = [int(float(s)) for s in sys.argv[4].split()]
rest = sys.argv[5:]
os.makedirs(outdir, exist_ok=True)
title = rest[rest.index('--title')+1] if '--title' in rest else 'Energy-norm RTM: shot images and common image gathers'
subtitle = rest[rest.index('--subtitle')+1] if '--subtitle' in rest else \
    'Rocha-geometry reflector, smooth migration model, %d shots' % len(shots)
show = [int(float(v)) for v in rest[rest.index('--show')+1].split()] if '--show' in rest else \
    [shots[int(round(i*(len(shots)-1)/4))] for i in range(5)] if len(shots) >= 5 else shots
figs = ['cig_atten.png', 'cig_full.png']
DX = DZ = 10.0; ZCLIP = 300.0
for kind in ['atten', 'full']:
    imgs = {xs: read_su(os.path.join(root, 'shot_x%d' % xs, 'reflector_abs_taper_refsub', 'energy_smooth_%s.su' % kind)) for xs in show}
    c = clipval(np.array(list(imgs.values())), int(ZCLIP/DZ))
    stk = sum(read_su(os.path.join(root, 'shot_x%d' % xs, 'reflector_abs_taper_refsub', 'energy_smooth_%s.su' % kind)) for xs in shots)
    panels = [('x$_s$=%d m' % xs, imgs[xs], c, [xs]) for xs in show] + \
             [('stack of %d shots' % len(shots), stk, clipval(stk, int(ZCLIP/DZ)), [])]
    ncol = 2; nrow = (len(panels)+ncol-1)//ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(13, 2.4*nrow+0.4), sharex=True, sharey=True, squeeze=False)
    for ax in axs.ravel()[len(panels):]: ax.axis('off')
    for ax, (lab, im, cl, src) in zip(axs.ravel(), panels):
        nx, nz = im.shape
        ax.imshow(im.T, aspect='auto', cmap='gray_r', vmin=-cl, vmax=cl, extent=[0, (nx-1)*DX, (nz-1)*DZ, 0])
        if src: ax.plot(src, [30], 'r*', ms=10, clip_on=False)
        ax.set_title(lab, fontsize=11)
    for ax in axs[:, 0]: ax.set_ylabel('depth (m)')
    for ax in axs[-1, :]: ax.set_xlabel('x (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'shots_%s.png' % kind), dpi=120); plt.close(fig)
for f in figs: shutil.copy(os.path.join(root, f), outdir)

dxs = sorted(set(b-a for a, b in zip(shots, shots[1:])))
spacing = ('%d\\,m' % dxs[0]) if len(dxs) == 1 else 'irregular'
L = []
L += [r'\documentclass[aspectratio=169]{beamer}', r'\usepackage{graphicx}',
      r'\setbeamertemplate{navigation symbols}{}',
      r'\title{%s}' % title, r'\subtitle{%s}' % subtitle, r'\author{}', r'\date{}', r'\begin{document}',
      r'\begin{frame}', r'\titlepage', r'\end{frame}', '']
L += [r'\begin{frame}{Setup}', r'\begin{itemize}',
      r'\item Model: $10000\times2500$\,m, $dx=dz=10$\,m, flat reflector at $z=1500$\,m (pure density contrast, $\rho$ 2000$\to$2500)',
      r'\item Sources: vertical force at $z=10$\,m, %d shots, $x_s=%d$--$%d$\,m every %s' % (len(shots), shots[0], shots[-1], spacing),
      r'\item Receiver line at $z=20$\,m, $x=10$--$9990$\,m (open line, cosine taper of 270 points at the ends), absorbing top',
      r'\item Scattered data by reference subtraction (REFSUB); reverse-time reconstruction and source wavefield both in the \textbf{smooth} reference model',
      r'\item Imaging condition: $I = \rho\,v_U\!\cdot\!v_V + \tau_U:S:\tau_V$ (full), \quad $I^\dagger = -\rho\,v_U\!\cdot\!v_V + \tau_U:S:\tau_V$ (attenuated)',
      r'\item Shot-index CIG at $x_0$: \quad $\mathrm{CIG}_{x_0}(z, x_s) = I_{x_s}(x_0, z)$, \quad $x_0 = %s$\,m' % r',\,'.join(str(x) for x in x0s),
      r'\end{itemize}', r'\end{frame}', '']
L += [r'\section{Individual shots}']
for kind, sym in [('atten', r'attenuated image $I^\dagger$'), ('full', r'full image $I$')]:
    L += [r'\begin{frame}{Shot images and stack, %s}' % sym,
          r'\centering\includegraphics[height=0.86\textheight]{shots_%s.png}' % kind, r'\end{frame}']
L += ['', r'\section{Common image gathers}']
for kind, sym in [('atten', r'attenuated image $I^\dagger$'), ('full', r'full image $I$')]:
    L += [r'\begin{frame}{Common image gathers, %s}' % sym,
          r'\centering\includegraphics[height=0.80\textheight]{cig_%s.png}\\' % kind,
          r'{\footnotesize Linear interpolation between shots along $x_s$ (display only).}', r'\end{frame}']
obs = os.path.join(outdir, 'observations.tex')
if os.path.exists(obs):
    L += [r'\begin{frame}{Observations}', r'\begin{itemize}', open(obs).read().strip(), r'\end{itemize}', r'\end{frame}']
L += [r'\end{document}']

base = rest[rest.index('--tex')+1] if '--tex' in rest else os.path.basename(os.path.normpath(outdir))
tex = base + '.tex'
open(os.path.join(outdir, tex), 'w').write('\n'.join(L) + '\n')
for _ in range(2):
    r = subprocess.run(['pdflatex', '-interaction=nonstopmode', tex], cwd=outdir, capture_output=True, text=True)
open(os.path.join(outdir, 'compile.log'), 'w').write(r.stdout)
errs = [l for l in r.stdout.splitlines() if l.startswith('!')]
print('pdflatex rc=%d%s -> %s' % (r.returncode, (' ERRORS: ' + '; '.join(errs)) if errs else '',
                                   os.path.join(outdir, base + '.pdf')))
sys.exit(r.returncode)
