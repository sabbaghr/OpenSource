#!/usr/bin/env python3
"""
wfl_splice_geo_hydro.py -- build a physically-realistic injection dataset from two separate
recordings of the SAME model/source, at two different receiver-line depths:

  geo_boundary_scat.bin    (z=505, in the solid)  -> keep vx, vz   ("geophone": particle velocity)
  hydro_boundary_scat.bin  (z=495, in the water)   -> keep tzz,txz ("hydrophone": tzz=-p, txz=0 there)

txx is dropped (zeroed): for a horizontal open line, injectBoundaryBwd.c's traction is T=(txz,tzz).n and
its stress-update terms use only vx,vz with the LOCAL medium parameters -- the recorded txx is never
read, so this loses nothing.

Output header/geometry (n, nt, xr, zr, nx, nz) is copied from the geophone (z=505) file, since that is
the line the reconstruction run's own geometry (wfl_z1=505) will independently regenerate and match
against.

  wfl_splice_geo_hydro.py geo_boundary_scat.bin hydro_boundary_scat.bin out.bin
"""
import struct, sys

def read(fn):
    d = open(fn, 'rb').read()
    n, nt = struct.unpack('<ii', d[:8])
    hdr = 8 + 4*4*n
    meta = d[8:hdr]
    names = ['vx','vz','txx','tzz','txz']
    blocks = {}
    off = hdr
    for nm in names:
        blocks[nm] = struct.unpack('<%df'%(n*nt), d[off:off+4*n*nt]); off += 4*n*nt
    return n, nt, meta, blocks

if __name__ == '__main__':
    geo_fn, hydro_fn, out_fn = sys.argv[1:4]
    ng, ntg, metag, geo = read(geo_fn)
    nh, nth, metah, hydro = read(hydro_fn)
    assert (ng, ntg) == (nh, nth), 'geometry mismatch: geo (%d,%d) vs hydro (%d,%d)'%(ng,ntg,nh,nth)

    out = bytearray()
    out += struct.pack('<ii', ng, ntg)
    out += metag  # xr,zr,nx,nz from the geophone (z=505) line -- this is the injection contour
    out += struct.pack('<%df'%(ng*ntg), *geo['vx'])
    out += struct.pack('<%df'%(ng*ntg), *geo['vz'])
    out += struct.pack('<%df'%(ng*ntg), *([0.0]*(ng*ntg)))   # txx: unused by the injection formula
    out += struct.pack('<%df'%(ng*ntg), *hydro['tzz'])
    out += struct.pack('<%df'%(ng*ntg), *hydro['txz'])
    open(out_fn, 'wb').write(out)

    # sanity check: confirm the hydrophone line really is in a fluid (txz~0, txx~tzz)
    import math
    rms_txz = math.sqrt(sum(v*v for v in hydro['txz'])/max(1,len(hydro['txz'])))
    rms_tzz = math.sqrt(sum(v*v for v in hydro['tzz'])/max(1,len(hydro['tzz'])))
    rms_diff = math.sqrt(sum((a-b)**2 for a,b in zip(hydro['txx'],hydro['tzz']))/max(1,len(hydro['tzz'])))
    print('wrote %s : vx,vz from %s ; tzz,txz from %s (%d points x %d samples)'%(out_fn, geo_fn, hydro_fn, ng, ntg))
    print('hydrophone-line sanity check (should look like a fluid): rms(txz)=%.3e (should be ~0 rel. to rms(tzz)=%.3e), rms(txx-tzz)=%.3e (should be ~0)'%(rms_txz, rms_tzz, rms_diff))
