#!/usr/bin/env python3
"""
wfl_rtm_poynting_integrated.py -- time-integrated Poynting-vector mask: accumulate the RAW Poynting
vectors (not angles, not per-step cos) for source and receiver fields over the whole shot duration,
then compute ONE stationary cos(theta) mask from the accumulated vectors, and apply it to the normal
(unweighted) total energy-norm image computed in the same pass.

    P_src(x,z) = sum_t S_src(x,z,t)     (vector sum, NOT per-step normalized)
    P_rec(x,z) = sum_t S_rec(x,z,t)
    cos(theta) = (P_src . P_rec) / (|P_src||P_rec| + eps)      -- computed ONCE, post-loop
    G(x,z)     = sum_t (kinetic_cross + potential_cross)        -- normal total energy image
    output     = G * (1-cos(theta))/2

  wfl_rtm_poynting_integrated.py nx nz dx dz case_dir model_prefix out_prefix [--src snap_ref]
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

    G = [[0.0]*nz for _ in range(nx)]
    Psrc_x = [[0.0]*nz for _ in range(nx)]; Psrc_z = [[0.0]*nz for _ in range(nx)]
    Prec_x = [[0.0]*nz for _ in range(nx)]; Prec_z = [[0.0]*nz for _ in range(nx)]
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
            muc, lamc, roc = mu[ix], lam[ix], ro[ix]
            gcol = G[ix]; psxc = Psrc_x[ix]; pszc = Psrc_z[ix]; prxc = Prec_x[ix]; przc = Prec_z[ix]
            for iz in range(nz):
                m = muc[iz]; l = lamc[iz]
                D = 4*m*(l+m)
                if D <= 0 or m <= 0:
                    continue
                exxV = ((l+2*m)*txxV[iz] - l*tzzV[iz]) / D
                ezzV = ((l+2*m)*tzzV[iz] - l*txxV[iz]) / D
                gxzV = txzV[iz] / m

                kinetic_cross = roc[iz]*(vxU[iz]*vxV[iz] + vzU[iz]*vzV[iz])
                potential_cross = txxU[iz]*exxV + tzzU[iz]*ezzV + txzU[iz]*gxzV
                gcol[iz] += kinetic_cross + potential_cross

                psxc[iz] += -(txxU[iz]*vxU[iz] + txzU[iz]*vzU[iz])
                pszc[iz] += -(txzU[iz]*vxU[iz] + tzzU[iz]*vzU[iz])
                prxc[iz] += -(txxV[iz]*vxV[iz] + txzV[iz]*vzV[iz])
                przc[iz] += -(txzV[iz]*vxV[iz] + tzzV[iz]*vzV[iz])
        if nfr % 10 == 0: print('frame', nfr, 'accumulated')
    print('%d frames accumulated (src=%s, model=%s)' % (nfr, src, model_prefix))

    eps = 1e-30
    cosmap = [[0.0]*nz for _ in range(nx)]
    out_img = [[0.0]*nz for _ in range(nx)]
    for ix in range(nx):
        for iz in range(nz):
            dot = Psrc_x[ix][iz]*Prec_x[ix][iz] + Psrc_z[ix][iz]*Prec_z[ix][iz]
            magS = (Psrc_x[ix][iz]**2 + Psrc_z[ix][iz]**2)**0.5
            magR = (Prec_x[ix][iz]**2 + Prec_z[ix][iz]**2)**0.5
            c = dot / (magS*magR + eps)
            cosmap[ix][iz] = c
            out_img[ix][iz] = G[ix][iz] * (1.0 - c) * 0.5

    write_su(out+'_G.su', nx, nz, G); png(out+'_G.png', nx, nz, G)
    write_su(out+'_cosmap_int.su', nx, nz, cosmap); png(out+'_cosmap_int.png', nx, nz, cosmap)
    write_su(out+'_masked_int.su', nx, nz, out_img); png(out+'_masked_int.png', nx, nz, out_img)
