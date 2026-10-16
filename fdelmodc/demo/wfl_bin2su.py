#!/usr/bin/env python3
"""
wfl_bin2su.py -- convert the recorded boundary/line wavefield (boundary.bin, written by fdelmodc with
wfl_recon=1) into SU gathers, one file per field, one trace per contour point:

    wfl_bin2su.py <boundary.bin> <dt> [outprefix=line]     ->  <outprefix>_r{vx,vz,txx,tzz,txz}.su

Trace order = contour order: for an open line left -> right; for the closed box top, bottom, left, right
(concatenated). Header: tracl = point number, ns, dt (us), gx = x (m), gelev = z (m).
These are exactly the data that are injected in the reconstruction run (raw fdelmodc fields, unnegated).

View, e.g. (open line, dx=4, x starts at wfl_x1):
    suximage < line_rvz.su perc=99 d1=<dt> d2=4 f2=<x1> xlabel="x (m)" ylabel="time (s)" title="recorded vz on the line"
    suxwigb  < line_rvz.su n2=<npoints> perc=99 x1beg=0 x1end=2 key=gx
"""
import struct, sys
fn=sys.argv[1]; dt=float(sys.argv[2]); pre=sys.argv[3] if len(sys.argv)>3 else 'line'
d=open(fn,'rb').read()
n,nt=struct.unpack('<ii',d[:8]); off=8
f=lambda k: struct.unpack('<%df'%k,d[off+0:off+4*k])
xr=struct.unpack('<%df'%n,d[off:off+4*n]);      off+=4*n
zr=struct.unpack('<%df'%n,d[off:off+4*n]);      off+=4*n
off+=8*n                                          # skip nx, nz
names=['vx','vz','txx','tzz','txz']
for name in names:
    blk=d[off:off+4*n*nt]; off+=4*n*nt
    out=bytearray()
    for i in range(n):
        h=bytearray(240)
        struct.pack_into('<i',h,0,i+1)            # tracl
        struct.pack_into('<i',h,4,i+1)            # tracr
        struct.pack_into('<h',h,70,1)             # scalco
        struct.pack_into('<i',h,80,int(round(xr[i])))   # gx
        struct.pack_into('<i',h,44,int(round(zr[i])))   # gelev
        struct.pack_into('<H',h,114,nt)           # ns
        struct.pack_into('<H',h,116,int(round(dt*1e6)))  # dt (us)
        out+=h; out+=blk[4*nt*i:4*nt*(i+1)]
    open('%s_r%s.su'%(pre,name),'wb').write(out)
print('wrote %s_r{%s}.su : %d traces x %d samples (dt=%g s), x = %.1f .. %.1f m, z = %.1f .. %.1f m'%(pre,','.join(names),n,nt,dt,min(xr),max(xr),min(zr),max(zr)))
