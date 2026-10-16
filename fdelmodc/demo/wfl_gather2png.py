#!/usr/bin/env python3
"""
wfl_gather2png.py -- render an SU gather (traces vs time, e.g. line_r*.su / line_scat_r*.su) to PNG,
so it can be viewed without X11/suximage.

  wfl_gather2png.py out.png file.su [--clip C | --perc P] [--tmax T]
"""
import sys, struct, array, zlib

def color(v, clip):
    t = max(-1.0, min(1.0, v/clip)) if clip else 0.0
    if t >= 0: return (255, int(255*(1-t)), int(255*(1-t)))
    return (int(255*(1+t)), int(255*(1+t)), 255)

if __name__ == '__main__':
    out, fn = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    perc = float(rest[rest.index('--perc')+1]) if '--perc' in rest else 99.0
    clip_arg = float(rest[rest.index('--clip')+1]) if '--clip' in rest else None
    tmax = float(rest[rest.index('--tmax')+1]) if '--tmax' in rest else None

    d = open(fn, 'rb').read()
    ns = struct.unpack('<H', d[114:116])[0]
    dt = struct.unpack('<H', d[116:118])[0] * 1e-6
    tr = 240 + 4*ns
    ntr = len(d)//tr
    ns_use = min(ns, int(tmax/dt)+1) if tmax else ns
    traces = []
    allvals = []
    for i in range(ntr):
        v = struct.unpack('<%df'%ns, d[i*tr+240:i*tr+240+4*ns])
        v = v[:ns_use]
        traces.append(v)
        allvals.extend(v)
    if clip_arg is not None:
        clip = clip_arg
    else:
        allvals = sorted(abs(x) for x in allvals)
        clip = allvals[int(perc/100.0*len(allvals))-1] if allvals else 1.0
    W, H = ntr, ns_use
    rows = [[255]*(3*W) for _ in range(H)]
    for ix in range(W):
        tr_v = traces[ix]
        for it in range(H):
            r,g,b = color(tr_v[it], clip)
            o = 3*ix
            rows[it][o]=r; rows[it][o+1]=g; rows[it][o+2]=b
    raw = b''.join(b'\x00'+bytes(r) for r in rows)
    def chunk(t,dd): return struct.pack('>I',len(dd))+t+dd+struct.pack('>I',zlib.crc32(t+dd)&0xffffffff)
    open(out,'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
    print('wrote', out, W, 'traces x', H, 'samples (dt=%g s, tmax=%.3f s), clip=%.3e'%(dt, H*dt, clip))
