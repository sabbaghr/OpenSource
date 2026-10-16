#!/usr/bin/env python3
"""
wfl_rtm_confidence.py -- directional + confidence-weighted imaging condition:

    W = C_U * C_V * (1-cos(theta))/2

where cos(theta) = (S_U.S_V)/(|S_U||S_V|) (same-direction test, as in wfl_rtm_poynting.py), and
C_U = |S_U| / (vp * E_U), C_V = |S_V| / (vp * E_V) are per-field "confidence" (plane-wave purity)
metrics: for a single coherent plane wave, |S| = c*E exactly (energy flux magnitude = energy density
x propagation speed), so C->1 for a clean single arrival and C->0 where multiple arrivals with
different directions are superposing at that point/time (their flux vectors partially cancel while
the energy densities, built from squared/product terms, keep adding). E_U, E_V here are each field's
own SELF energy (rho|v|^2 + tau:S:tau), not the U-V cross term.

Final image: sum_t (kinetic_cross + potential_cross) * W

  wfl_rtm_confidence.py nx nz dx dz case_dir model_prefix out_prefix [--src snap_ref]
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

    img_w = [[0.0]*nz for _ in range(nx)]
    cu_acc = [[0.0]*nz for _ in range(nx)]
    cv_acc = [[0.0]*nz for _ in range(nx)]
    w_acc  = [[0.0]*nz for _ in range(nx)]
    nfr = 0
    eps = 1e-30
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
            muc, lamc, cpc, roc = mu[ix], lam[ix], cp[ix], ro[ix]
            wcol, cucol, cvcol, wacol = img_w[ix], cu_acc[ix], cv_acc[ix], w_acc[ix]
            for iz in range(nz):
                m = muc[iz]; l = lamc[iz]
                D = 4*m*(l+m)
                if D <= 0 or m <= 0:
                    continue
                exxU = ((l+2*m)*txxU[iz] - l*tzzU[iz]) / D
                ezzU = ((l+2*m)*tzzU[iz] - l*txxU[iz]) / D
                gxzU = txzU[iz] / m
                exxV = ((l+2*m)*txxV[iz] - l*tzzV[iz]) / D
                ezzV = ((l+2*m)*tzzV[iz] - l*txxV[iz]) / D
                gxzV = txzV[iz] / m

                rho = roc[iz]
                kinetic_cross = rho*(vxU[iz]*vxV[iz] + vzU[iz]*vzV[iz])
                potential_cross = txxU[iz]*exxV + tzzU[iz]*ezzV + txzU[iz]*gxzV
                energy_cross = kinetic_cross + potential_cross

                E_U = rho*(vxU[iz]*vxU[iz]+vzU[iz]*vzU[iz]) + (txxU[iz]*exxU+tzzU[iz]*ezzU+txzU[iz]*gxzU)
                E_V = rho*(vxV[iz]*vxV[iz]+vzV[iz]*vzV[iz]) + (txxV[iz]*exxV+tzzV[iz]*ezzV+txzV[iz]*gxzV)

                SUx = -(txxU[iz]*vxU[iz] + txzU[iz]*vzU[iz])
                SUz = -(txzU[iz]*vxU[iz] + tzzU[iz]*vzU[iz])
                SVx = -(txxV[iz]*vxV[iz] + txzV[iz]*vzV[iz])
                SVz = -(txzV[iz]*vxV[iz] + tzzV[iz]*vzV[iz])
                magU = (SUx*SUx + SUz*SUz)**0.5
                magV = (SVx*SVx + SVz*SVz)**0.5
                dot = SUx*SVx + SUz*SVz
                cos_theta = dot / (magU*magV + eps)

                vp = cpc[iz]
                C_U = magU / (vp*abs(E_U) + eps)
                C_V = magV / (vp*abs(E_V) + eps)
                C_U = min(C_U, 1.0); C_V = min(C_V, 1.0)   # clip numerical overshoot

                weight = C_U * C_V * (1.0 - cos_theta) * 0.5

                wcol[iz] += energy_cross * weight
                cucol[iz] += C_U
                cvcol[iz] += C_V
                wacol[iz] += weight
        if nfr % 10 == 0: print('frame', nfr, 'accumulated')
    print('%d frames accumulated (src=%s, model=%s)' % (nfr, src, model_prefix))

    cu_avg = [[cu_acc[ix][iz]/nfr for iz in range(nz)] for ix in range(nx)]
    cv_avg = [[cv_acc[ix][iz]/nfr for iz in range(nz)] for ix in range(nx)]
    w_avg  = [[w_acc[ix][iz]/nfr for iz in range(nz)] for ix in range(nx)]

    write_su(out+'_confimg.su', nx, nz, img_w); png(out+'_confimg.png', nx, nz, img_w)
    write_su(out+'_Cu.su', nx, nz, cu_avg); png(out+'_Cu.png', nx, nz, cu_avg)
    write_su(out+'_Cv.su', nx, nz, cv_avg); png(out+'_Cv.png', nx, nz, cv_avg)
    write_su(out+'_weight.su', nx, nz, w_avg); png(out+'_weight.png', nx, nz, w_avg)
