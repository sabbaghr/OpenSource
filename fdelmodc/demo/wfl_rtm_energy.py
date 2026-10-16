#!/usr/bin/env python3
"""
wfl_rtm_energy.py -- energy-norm (energy inner product) imaging condition, as opposed to the plain
dot-product imaging condition in wfl_rtm.py.

Uses the SAME bilinear pairing that makes the elastodynamic operator exactly skew-adjoint for
heterogeneous media (representation-theorem-mps-adjoint.pdf, Thm 7.3 / eq. 25):

    <w1,w2>_E = int rho(x) v1.v2 dx  +  int s_ijkl(x) tau1_ij tau2_kl dx

i.e. kinetic energy (density-weighted velocity dot product) PLUS strain energy (compliance-weighted
stress contraction). The strain-energy compliance is inverted using sum/difference stress invariants
S=txx+tzz (isotropic part, well-behaved everywhere) and D=txx-tzz (deviatoric part, paired with the
shear stress txz), which cleanly isolates the shear modulus mu in the denominator ONLY where it
actually matters physically (a fluid, mu=0, truly has zero deviatoric stress/strain, so that term is
just set to 0 there rather than evaluating a 0/0):

    I(x,z) = sum_t [ rho*(vx_s*vx_r + vz_s*vz_r)                                 (kinetic)
                     + Ss*Sr/(4*(lam+mu))                                        (isotropic/bulk strain)
                     + [Ds*Dr/(4*mu) + txz_s*txz_r/mu]  (only where mu > threshold)  (deviatoric/shear strain)
                   ]

where S=txx+tzz, D=txx-tzz, and rho, lam=rho(cp^2-2cs^2), mu=rho*cs^2 are the RAW elastic parameters
(from model_cp.su/model_cs.su/model_ro.su, NOT the FD-internal dt/dx-scaled rox/l2m/etc.), sampled at
each image point; s=source-side (e.g. snap_ref), r=receiver-side (e.g. snap_recon_trev).

  wfl_rtm_energy.py sizex sizez dx dz case_dir out_prefix src_prefix [--zmin Z --zmax Z]

Requires, in case_dir: modelref_cp.su, modelref_cs.su, modelref_ro.su (the MIGRATION/background model --
the elastic parameters must be the ones the wavefields were actually propagated through, i.e. the same
smooth model used for both the source-side reference run and the receiver-side reconstruction, not the
true model), and snap_<src_prefix>_s{vx,vz,txx,tzz,txz}.su plus snap_recon_trev_s{vx,vz,txx,tzz,txz}.su
-- all five fields, both sides.
"""
import sys, struct, array, os, math

def read_model(fn, nx, nz):
    tr = 240 + 4*nz
    with open(fn,'rb') as f:
        b = f.read(tr*nx)
    out = []
    for ix in range(nx):
        a = array.array('f'); a.frombytes(b[ix*tr+240:(ix+1)*tr]); out.append(a)
    return out

def frames(fn, nx, nz):
    tr = 240 + 4*nz
    n = os.path.getsize(fn)//(tr*nx)
    with open(fn,'rb') as f:
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
        struct.pack_into('<i',h,0,ix+1); struct.pack_into('<i',h,4,ix+1)
        struct.pack_into('<H',h,114,nz); struct.pack_into('<H',h,116,1)
        out += h; out += struct.pack('<%df'%nz, *img[ix])
    open(fn,'wb').write(out)

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
            v = sum(img[ix+ddx][iz+ddz] for ddx in range(scale) for ddz in range(scale))/(scale*scale)
            r,g,b = color(v, clip); o = 3*(ix//scale)
            rows[iz//scale][o]=r; rows[iz//scale][o+1]=g; rows[iz//scale][o+2]=b
    raw = b''.join(b'\x00'+bytes(r) for r in rows)
    import zlib
    def chunk(t,dd): return struct.pack('>I',len(dd))+t+dd+struct.pack('>I',zlib.crc32(t+dd)&0xffffffff)
    open(fn,'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
    print('wrote', fn, W, 'x', H, 'clip=%.3e'%clip)

if __name__ == '__main__':
    sizex, sizez, dx, dz = [float(v) for v in sys.argv[1:5]]
    case_dir, out, src = sys.argv[5], sys.argv[6], sys.argv[7]
    rest = sys.argv[8:]
    zmin = float(rest[rest.index('--zmin')+1]) if '--zmin' in rest else None
    zmax = float(rest[rest.index('--zmax')+1]) if '--zmax' in rest else None
    kmin = int(rest[rest.index('--kmin')+1]) if '--kmin' in rest else None
    kmax = int(rest[rest.index('--kmax')+1]) if '--kmax' in rest else None
    nx = int(round(sizex/dx))+1; nz = int(round(sizez/dz))+1
    iz0 = int(round(zmin/dz)) if zmin is not None else 0
    iz1 = int(round(zmax/dz))+1 if zmax is not None else nz

    # migration/background model (modelref_*), NOT the true model -- the energy weighting must match
    # the medium the wavefields were actually propagated through
    cp = read_model(os.path.join(case_dir,'modelref_cp.su'), nx, nz)
    cs = read_model(os.path.join(case_dir,'modelref_cs.su'), nx, nz)
    ro = read_model(os.path.join(case_dir,'modelref_ro.su'), nx, nz)
    # raw (unscaled) elastic parameters at every grid point
    lam = [[ro[ix][iz]*(cp[ix][iz]**2 - 2*cs[ix][iz]**2) for iz in range(nz)] for ix in range(nx)]
    mu  = [[ro[ix][iz]*cs[ix][iz]**2 for iz in range(nz)] for ix in range(nx)]
    mu_max = max(max(row) for row in mu)
    mu_thresh = mu_max * 1e-6  # below this, treat the medium as a fluid: deviatoric/shear strain energy = 0

    fields = ['vx','vz','txx','tzz','txz']
    src_fn  = {f: os.path.join(case_dir, 'snap_%s_s%s.su'%(src,f)) for f in fields}
    recv_fn = {f: os.path.join(case_dir, 'snap_recon_trev_s%s.su'%f) for f in fields}
    for f in fields:
        assert os.path.exists(src_fn[f]), 'missing %s (need all 5 fields for the energy norm)'%src_fn[f]
        assert os.path.exists(recv_fn[f]), 'missing %s (need all 5 fields for the energy norm)'%recv_fn[f]

    gens = {f: zip(frames(src_fn[f], nx, nz), frames(recv_fn[f], nx, nz)) for f in fields}
    img = [[0.0]*nz for _ in range(nx)]
    nfr = 0
    k = -1
    while True:
        k += 1
        try:
            fr = {f: next(gens[f]) for f in fields}
        except StopIteration:
            break
        if kmin is not None and k < kmin: continue
        if kmax is not None and k > kmax: continue
        nfr += 1
        for ix in range(nx):
            vxs,vxr = fr['vx'][0][ix], fr['vx'][1][ix]
            vzs,vzr = fr['vz'][0][ix], fr['vz'][1][ix]
            txxs,txxr = fr['txx'][0][ix], fr['txx'][1][ix]
            tzzs,tzzr = fr['tzz'][0][ix], fr['tzz'][1][ix]
            txzs,txzr = fr['txz'][0][ix], fr['txz'][1][ix]
            im = img[ix]
            for iz in range(iz0, iz1):
                kinetic = ro[ix][iz]*(vxs[iz]*vxr[iz] + vzs[iz]*vzr[iz])
                Ss = txxs[iz]+tzzs[iz]; Sr = txxr[iz]+tzzr[iz]
                isotropic = Ss*Sr/(4.0*(lam[ix][iz]+mu[ix][iz]))
                m = mu[ix][iz]
                if m > mu_thresh:
                    Ds = txxs[iz]-tzzs[iz]; Dr = txxr[iz]-tzzr[iz]
                    deviatoric = Ds*Dr/(4.0*m) + txzs[iz]*txzr[iz]/m
                else:
                    deviatoric = 0.0  # fluid: no shear stiffness to store energy in this mode
                im[iz] += kinetic + isotropic + deviatoric
    print('accumulated %d frames (k=%s..%s)'%(nfr, kmin if kmin is not None else 0, kmax if kmax is not None else 'end'))
    write_su(out+'.su', nx, nz, img)
    png(out+'.png', nx, nz, img)
