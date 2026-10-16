#!/usr/bin/env python3
"""
wfl_rtm_energy_stress.py -- Rocha, Tanushev & Sava (2016) energy-norm imaging condition, reformulated
in terms of PARTICLE VELOCITY and STRESS (fdelmodc's native fields) instead of displacement and its
gradient.

Kinetic term: U_dot . V_dot  ==  vx,vz directly (velocity IS displacement-rate already).

Potential term: Hooke's law inverted to recover strain from recorded stress, using the migration
model's own lambda,mu(x,z):
    exx = [(lam+2mu)*txx - lam*tzz] / D         D = 4*mu*(lam+mu)
    ezz = [(lam+2mu)*tzz - lam*txx] / D
    gxz = txz / mu
Dilatation (div U) term:  div U = (txx+tzz) / (2*(lam+mu))
    (vp^2-vs^2)*(div U)(div V)  ==  (txx_U+tzz_U)(txx_V+tzz_V) / (4*rho*(lam+mu))

Shear term: Rocha's eq. 21/23 uses the BARE gradient tensor grad(U), which includes a rotational part
that stress cannot see (stress only constrains symmetric strain). We substitute the ROTATION-FREE
symmetric-strain bilinear form, which is exactly recoverable from stress and is the form that appears
in their own CONSERVED energy (eq. 15, where the rotational part exactly cancels):
    vs^2 * [exx_U*exx_V + ezz_U*ezz_V + 0.5*gxz_U*gxz_V]

Full image:      I_E (x,z)  = sum_t  [ vx_U*vx_V + vz_U*vz_V  +  dilatation term  +  shear term ]
Attenuated image I_E_dag    = sum_t  [ -vx_U*vx_V - vz_U*vz_V +  dilatation term  +  shear term ]
  (eq. 39: (square V)^dagger flips the sign of V_dot only)

  wfl_rtm_energy_stress.py nx nz dx dz case_dir model_prefix out_prefix [--src snap_ref]

model_prefix: e.g. 'model' (uses model_cp.su model_cs.su model_ro.su) or 'modelref'.
--src: source-side snapshot prefix (default snap_forw; use snap_ref for REFSUB-consistent imaging).
Reads snap_<src>_s{vx,vz,txx,tzz,txz}.su and snap_recon_trev_s{vx,vz,txx,tzz,txz}.su from case_dir.
Writes <out_prefix>_full.su/.png and <out_prefix>_atten.su/.png.
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

    img_full  = [[0.0]*nz for _ in range(nx)]
    img_atten = [[0.0]*nz for _ in range(nx)]
    img_kin   = [[0.0]*nz for _ in range(nx)]
    img_pot   = [[0.0]*nz for _ in range(nx)]
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
            fcol, acol, kcol, pcol = img_full[ix], img_atten[ix], img_kin[ix], img_pot[ix]
            for iz in range(nz):
                m = muc[iz]; l = lamc[iz]
                D = 4*m*(l+m)
                if D <= 0 or m <= 0:
                    continue
                # S:tau (compliance contracted with stress) -- exx,ezz,gxz ARE the strain recovered
                # from stress via the compliance tensor S=C^-1; this is exact, no displacement needed.
                exxV = ((l+2*m)*txxV[iz] - l*tzzV[iz]) / D
                ezzV = ((l+2*m)*tzzV[iz] - l*txxV[iz]) / D
                gxzV = txzV[iz] / m

                # E = 1/2 integral( rho*v_i*v_i + tau_ij S_ijkl tau_kl ) dV  -- the exact elastic energy
                # in velocity-stress variables (user-specified form). tau_U : S : tau_V = tau_U : eps_V
                # (compliance is symmetric, so this is already the correct U-V bilinear form, no extra
                # symmetrization needed).
                kinetic = ro[ix][iz]*(vxU[iz]*vxV[iz] + vzU[iz]*vzV[iz])
                potential = txxU[iz]*exxV + tzzU[iz]*ezzV + txzU[iz]*gxzV

                fcol[iz] += kinetic + potential
                acol[iz] += -kinetic + potential
                kcol[iz] += kinetic
                pcol[iz] += potential
        if nfr % 10 == 0: print('frame', nfr, 'accumulated')
    print('%d frames accumulated (src=%s, model=%s)' % (nfr, src, model_prefix))

    write_su(out+'_full.su', nx, nz, img_full); png(out+'_full.png', nx, nz, img_full)
    write_su(out+'_atten.su', nx, nz, img_atten); png(out+'_atten.png', nx, nz, img_atten)
    write_su(out+'_kinetic.su', nx, nz, img_kin); png(out+'_kinetic.png', nx, nz, img_kin)
    write_su(out+'_potential.su', nx, nz, img_pot); png(out+'_potential.png', nx, nz, img_pot)
