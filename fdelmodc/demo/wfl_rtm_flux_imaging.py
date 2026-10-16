#!/usr/bin/env python3
"""
wfl_rtm_flux_imaging.py -- energy FLUX imaging condition: sum_t S_U(x,z,t) . S_V(x,z,t), where
S = +tau.v (Poynting/energy-flux vector, sign-corrected for fdelmodc's stress convention -- verified
directly: below a source, tzz and vz are in phase for an outgoing wave, requiring S=+tau.v to give
Sz>0 there). This is distinct from the kinetic+potential energy-norm imaging condition: it correlates
the two fields' FLUX vectors directly, rather than correlating velocity-with-velocity and
stress-with-stress separately and summing.

  wfl_rtm_flux_imaging.py nx nz dx dz case_dir out_prefix [--src snap_ref]
"""
import sys, struct, array, os

def frames(fn, nx, nz):
    tr = 240 + 4*nz
    n = os.path.getsize(fn)//(tr*nx)
    with open(fn, 'rb') as f:
        for k in range(n):
            b = f.read(tr*nx)
            cols = []
            for ix in range(nx):
                a = array.array('f'); a.frombytes(b[ix*tr+240:(ix+1)*tr]); cols.append(a)
            yield cols

def write_su(fn, nx, nz, img):
    tr = 240 + 4*nz
    out = bytearray()
    for ix in range(nx):
        h = bytearray(240)
        struct.pack_into('<i', h, 0, ix+1); struct.pack_into('<i', h, 4, ix+1)
        struct.pack_into('<H', h, 114, nz); struct.pack_into('<H', h, 116, 1)
        out += h; out += struct.pack('<%df'%nz, *img[ix])
    open(fn, 'wb').write(out)

def png(fn, nx, nz, img, scale=2):
    def color(v, clip):
        t = max(-1.0, min(1.0, v/clip)) if clip else 0.0
        if t >= 0: return (255, int(255*(1-t)), int(255*(1-t)))
        return (int(255*(1+t)), int(255*(1+t)), 255)
    vals = sorted(abs(v) for col in img for v in col)
    clip = vals[int(0.98*len(vals))] if vals else 1.0
    W = nx//scale; H = nz//scale
    rows = [[255]*(3*W) for _ in range(H)]
    for ix in range(0, nx-scale+1, scale):
        col = img[ix]
        for iz in range(0, nz-scale+1, scale):
            v = sum(img[ix+dx_][iz+dz_] for dx_ in range(scale) for dz_ in range(scale))/(scale*scale)
            r, g, b = color(v, clip); o = 3*(ix//scale)
            rows[iz//scale][o] = r; rows[iz//scale][o+1] = g; rows[iz//scale][o+2] = b
    raw = b''.join(b'\x00'+bytes(r) for r in rows)
    import zlib
    def chunk(t, dd): return struct.pack('>I', len(dd))+t+dd+struct.pack('>I', zlib.crc32(t+dd)&0xffffffff)
    open(fn, 'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))+chunk(b'IDAT', zlib.compress(raw, 6))+chunk(b'IEND', b''))
    print('wrote', fn, W, 'x', H, 'clip=%.3e'%clip)

if __name__ == '__main__':
    nx, nz = int(sys.argv[1]), int(sys.argv[2])
    dx, dz = float(sys.argv[3]), float(sys.argv[4])
    case_dir, out = sys.argv[5], sys.argv[6]
    rest = sys.argv[7:]
    src = 'snap_forw'
    if '--src' in rest: src = rest[rest.index('--src')+1]

    def fn(prefix, field): return os.path.join(case_dir, '%s_s%s.su'%(prefix, field))
    gens = {f: frames(fn(src, f), nx, nz) for f in ['vx','vz','txx','tzz','txz']}
    gens_r = {f: frames(fn('snap_recon_trev', f), nx, nz) for f in ['vx','vz','txx','tzz','txz']}

    img = [[0.0]*nz for _ in range(nx)]
    nfr = 0
    while True:
        try:
            U = {f: next(gens[f]) for f in gens}
            V = {f: next(gens_r[f]) for f in gens_r}
        except StopIteration:
            break
        nfr += 1
        for ix in range(nx):
            txxU, tzzU, txzU = U['txx'][ix], U['tzz'][ix], U['txz'][ix]
            txxV, tzzV, txzV = V['txx'][ix], V['tzz'][ix], V['txz'][ix]
            vxU, vzU = U['vx'][ix], U['vz'][ix]
            vxV, vzV = V['vx'][ix], V['vz'][ix]
            col = img[ix]
            for iz in range(nz):
                SUx = txxU[iz]*vxU[iz] + txzU[iz]*vzU[iz]
                SUz = txzU[iz]*vxU[iz] + tzzU[iz]*vzU[iz]
                SVx = txxV[iz]*vxV[iz] + txzV[iz]*vzV[iz]
                SVz = txzV[iz]*vxV[iz] + tzzV[iz]*vzV[iz]
                col[iz] += SUx*SVx + SUz*SVz
        if nfr % 10 == 0: print('frame', nfr, 'accumulated')
    print('%d frames accumulated (src=%s)' % (nfr, src))

    write_su(out+'_flux.su', nx, nz, img); png(out+'_flux.png', nx, nz, img)
