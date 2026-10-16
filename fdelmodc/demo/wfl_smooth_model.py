#!/usr/bin/env python3
"""
wfl_smooth_model.py -- smooth an SU model file below a given depth, leaving everything above it
(e.g. the water column and the seafloor itself) exactly untouched.

  wfl_smooth_model.py file.su dz z0 halfwidth_m [--niter N]

Smooths samples with iz >= z0/dz using a boxcar of full width 2*halfwidth_m (in metres, converted to
samples via dz), per trace, clamped to [iz0, nz-1] (the kernel never reaches into, or even reads from,
the z<z0 part, so the water layer / seafloor boundary is exactly preserved -- only the deeper part of
the model is blurred). Overwrites the file in place.

--niter N (default 1): apply the boxcar N times (each pass re-clamped to [iz0, nz-1]) instead of once.
By the Central Limit Theorem, repeated boxcar convolution approximates a GAUSSIAN kernel -- a smooth,
tapered onset with no sharp derivative discontinuity, unlike a single hard-edged boxcar (N=1). This is
the same technique as OpenSource's own `smooth` utility (utils/smooth.c), just re-implemented here with
the z0 protection that utility doesn't have (it would smooth the whole trace, including the shallow
known structure we want kept exact).
"""
import struct, sys

def boxcar_pass(v, iz0, ns, hwn):
    sm = v[:]
    for iz in range(iz0, ns):
        a = max(iz0, iz-hwn); b = min(ns, iz+hwn+1)
        sm[iz] = sum(v[a:b])/(b-a)
    return sm

if __name__ == '__main__':
    fn, dz, z0, hw = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    rest = sys.argv[5:]
    niter = int(rest[rest.index('--niter')+1]) if '--niter' in rest else 1
    d = bytearray(open(fn, 'rb').read())
    ns = struct.unpack('<H', d[114:116])[0]
    tr = 240 + 4*ns
    ntr = len(d)//tr
    iz0 = int(round(z0/dz))
    hwn = max(1, int(round(hw/dz)))
    for i in range(ntr):
        off = i*tr + 240
        v = list(struct.unpack('<%df'%ns, d[off:off+4*ns]))
        for _ in range(niter):
            v = boxcar_pass(v, iz0, ns, hwn)
        struct.pack_into('<%df'%ns, d, off, *v)
    open(fn, 'wb').write(d)
    print('%s: smoothed z>=%.0f m (boxcar half-width %.0f m = %d samples, niter=%d), %d traces'%(fn, z0, hw, hwn, niter, ntr))
