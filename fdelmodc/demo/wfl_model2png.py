#!/usr/bin/env python3
"""
wfl_model2png.py -- render one or more SU model files (single frame, e.g. model_cp.su, modelref_cp.su)
side by side as PNG, using a sequential (min-to-max) colormap -- NOT the diverging red/blue scheme used
for wavefields, since model values (velocity, density) don't straddle zero.

  wfl_model2png.py out.png nx nz vmin vmax file1.su [file2.su ...] [--scale N]
"""
import sys, struct, array, zlib

def color(v, vmin, vmax):
    t = max(0.0, min(1.0, (v-vmin)/(vmax-vmin))) if vmax>vmin else 0.5
    # simple blue (low) -> white (mid) -> red (high) sequential-ish map, but monotonic in brightness too
    r = int(255*min(1.0, 2*t))
    b = int(255*min(1.0, 2*(1-t)))
    g = int(255*(1-abs(2*t-1))*0.6 + 255*0.0)
    return (r, g, b)

if __name__ == '__main__':
    out = sys.argv[1]; nx=int(sys.argv[2]); nz=int(sys.argv[3])
    vmin=float(sys.argv[4]); vmax=float(sys.argv[5])
    rest = sys.argv[6:]
    scale = int(rest[rest.index('--scale')+1]) if '--scale' in rest else 2
    files = [a for a in rest if a!='--scale' and not a.isdigit()]
    tr = 240+4*nz
    panels=[]
    for fn in files:
        d=open(fn,'rb').read()
        cols=[]
        for ix in range(nx):
            a=array.array('f'); a.frombytes(d[ix*tr+240:(ix+1)*tr]); cols.append(a)
        panels.append(cols)
    gap=6
    W=len(panels)*(nx//scale)+(len(panels)-1)*gap; H=nz//scale
    rows=[[255]*(3*W) for _ in range(H)]
    for p,cols in enumerate(panels):
        x0=p*(nx//scale+gap)
        for ix in range(0,nx-scale+1,scale):
            col=cols[ix]
            for iz in range(0,nz-scale+1,scale):
                v=sum(cols[ix+ddx][iz+ddz] for ddx in range(scale) for ddz in range(scale))/(scale*scale)
                r,g,b=color(v,vmin,vmax); o=3*(x0+ix//scale)
                rows[iz//scale][o]=r; rows[iz//scale][o+1]=g; rows[iz//scale][o+2]=b
    raw=b''.join(b'\x00'+bytes(r) for r in rows)
    def chunk(t,dd): return struct.pack('>I',len(dd))+t+dd+struct.pack('>I',zlib.crc32(t+dd)&0xffffffff)
    open(out,'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
    print('wrote',out,W,'x',H,'range',vmin,vmax)
