#!/usr/bin/env python3
"""
wfl_rtm_poynting.py -- use the elastic Poynting (energy-flux) vector S_i = -tau_ij v_j to directly
test the backscattering condition (source and receiver wavefields traveling the same direction, i.e.
sharing the same slowness vector p -- the literal geometric condition behind the Christoffel-equation
cancellation used by the modified energy norm, eq. 39), instead of relying on the kinetic/potential
sign-flip trick alone.

At each (x,z,t):
    S_U = (-(txxU*vxU+txzU*vzU), -(txzU*vxU+tzzU*vzU))   -- source-side energy flux
    S_V = (-(txxV*vxV+txzV*vzV), -(txzV*vxV+tzzV*vzV))   -- receiver-side energy flux
    cos(theta) = (S_U . S_V) / (|S_U||S_V|)              -- +1 same direction (backscatter), -1 opposite

Outputs:
  <out>_cosmap.su/.png   -- energy-weighted time-average of cos(theta): sum_t(S_U.S_V) / sum_t(|S_U||S_V|)
  <out>_dirimg.su/.png   -- directionally-weighted image: sum_t (kinetic+potential) * (1-cos(theta))/2
                            (0 weight when same-direction/backscatter, 1 when opposite, 0.5 perpendicular)

  wfl_rtm_poynting.py nx nz dx dz case_dir model_prefix out_prefix [--src snap_ref]
"""
import sys, struct, array, os

def read_model(fn, nx, nz):
    tr = 240 + 4*nz
    out = [None]*nx
    with open(fn, 'rb') as f:
        for ix in range(nx):
            b = f.read(tr)
            a = array.array('f'); a.frombytes(b[240:])
            out[ix] = a
    return out

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
    case_dir, model_prefix, out = sys.argv[5], sys.argv[6], sys.argv[7]
    rest = sys.argv[8:]
    src = 'snap_forw'
    if '--src' in rest: src = rest[rest.index('--src')+1]

    cp = read_model(os.path.join(case_dir, model_prefix+'_cp.su'), nx, nz)
    cs = read_model(os.path.join(case_dir, model_prefix+'_cs.su'), nx, nz)
    ro = read_model(os.path.join(case_dir, model_prefix+'_ro.su'), nx, nz)
    mu  = [[ro[ix][iz]*cs[ix][iz]**2 for iz in range(nz)] for ix in range(nx)]
    lam = [[ro[ix][iz]*(cp[ix][iz]**2 - 2*cs[ix][iz]**2) for iz in range(nz)] for ix in range(nx)]

    def fn(prefix, field): return os.path.join(case_dir, '%s_s%s.su'%(prefix, field))
    gens = {f: frames(fn(src, f), nx, nz) for f in ['vx','vz','txx','tzz','txz']}
    gens_r = {f: frames(fn('snap_recon_trev', f), nx, nz) for f in ['vx','vz','txx','tzz','txz']}

    img_dir = [[0.0]*nz for _ in range(nx)]
    cos_num = [[0.0]*nz for _ in range(nx)]
    cos_den = [[0.0]*nz for _ in range(nx)]
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
            muc, lamc = mu[ix], lam[ix]
            dcol, ncol, dencol = img_dir[ix], cos_num[ix], cos_den[ix]
            for iz in range(nz):
                m = muc[iz]; l = lamc[iz]
                D = 4*m*(l+m)
                if D <= 0 or m <= 0:
                    continue
                exxV = ((l+2*m)*txxV[iz] - l*tzzV[iz]) / D
                ezzV = ((l+2*m)*tzzV[iz] - l*txxV[iz]) / D
                gxzV = txzV[iz] / m

                kinetic = ro[ix][iz]*(vxU[iz]*vxV[iz] + vzU[iz]*vzV[iz])
                potential = txxU[iz]*exxV + tzzU[iz]*ezzV + txzU[iz]*gxzV
                energy = kinetic + potential

                SUx = -(txxU[iz]*vxU[iz] + txzU[iz]*vzU[iz])
                SUz = -(txzU[iz]*vxU[iz] + tzzU[iz]*vzU[iz])
                SVx = -(txxV[iz]*vxV[iz] + txzV[iz]*vzV[iz])
                SVz = -(txzV[iz]*vxV[iz] + tzzV[iz]*vzV[iz])
                dot = SUx*SVx + SUz*SVz
                magU = (SUx*SUx + SUz*SUz)**0.5
                magV = (SVx*SVx + SVz*SVz)**0.5
                magprod = magU*magV
                eps = 1e-30
                cos_theta = dot / (magprod + eps)
                weight = (1.0 - cos_theta) * 0.5

                dcol[iz] += energy * weight
                ncol[iz] += dot
                dencol[iz] += magprod
        if nfr % 10 == 0: print('frame', nfr, 'accumulated')
    print('%d frames accumulated (src=%s, model=%s)' % (nfr, src, model_prefix))

    cosmap = [[ (cos_num[ix][iz] / (cos_den[ix][iz] + 1e-30)) for iz in range(nz)] for ix in range(nx)]

    write_su(out+'_dirimg.su', nx, nz, img_dir); png(out+'_dirimg.png', nx, nz, img_dir)
    write_su(out+'_cosmap.su', nx, nz, cosmap); png(out+'_cosmap.png', nx, nz, cosmap)
