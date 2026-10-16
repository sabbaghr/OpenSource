#!/usr/bin/env python3
"""
wfl_rtm_nonlinear_velocity.py -- linear and nonlinear source-receiver imaging conditions
(Ravasi, Vasconcelos & Curtis 2014, GJI 198:1173-1185, eq. 3), reformulated for particle
velocity (eq. 1's I_ji = G^{S(v,f)}_{(i,j)}(x,x,t=0) form). UNLIKE the energy-norm imaging
condition (wfl_rtm_energy_stress.py), this is a PLAIN crosscorrelation of velocity components
-- no density/stiffness-tensor weighting beyond the constant reference-model scale 2/(rho0*cp0).

Linear term:    I_lin(x,z) = (2/(rho0*cp0)) sum_t [ vxU.vxV + vzU.vzV ]
                U = snap_ref (source-side, REFERENCE/background forward field)
                V = snap_recon_trev from the REFSUB=1 case (receiver-side, back-propagated
                    SCATTERED-only data)

Nonlinear term: I_nl(x,z)  = (2/(rho0*cp0)) sum_t [ vxU'.vxV' + vzU'.vzV' ]
                U' = snap_forw - snap_ref (source-side SCATTERED field, true minus reference)
                V' = snap_recon_trev from the REFSUB=0 case (receiver-side, back-propagated
                     TOTAL/raw recorded data)

Full nonlinear image: I_full = I_lin + I_nl  (eq. 3's two integrals summed)

  wfl_rtm_nonlinear_velocity.py nx nz dx dz lin_dir nl_recv_dir out_prefix rho0 cp0

lin_dir:     case directory holding snap_forw, snap_ref, and the REFSUB=1 snap_recon_trev
nl_recv_dir: case directory holding the REFSUB=0 snap_recon_trev (receiver-side, total data)
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

def crosscorrelate(fnU_vx, fnU_vz, fnV_vx, fnV_vz, nx, nz, scale, subtract_ref=None):
    """subtract_ref: if given, (fn_ref_vx, fn_ref_vz) to subtract from U frame-by-frame
    (builds the source-side SCATTERED field U'=snap_forw-snap_ref on the fly)."""
    gU_vx = frames(fnU_vx, nx, nz); gU_vz = frames(fnU_vz, nx, nz)
    gV_vx = frames(fnV_vx, nx, nz); gV_vz = frames(fnV_vz, nx, nz)
    if subtract_ref:
        gR_vx = frames(subtract_ref[0], nx, nz); gR_vz = frames(subtract_ref[1], nx, nz)
    img = [[0.0]*nz for _ in range(nx)]
    nfr = 0
    while True:
        try:
            uvx = next(gU_vx); uvz = next(gU_vz)
            vvx = next(gV_vx); vvz = next(gV_vz)
            if subtract_ref:
                rvx = next(gR_vx); rvz = next(gR_vz)
        except StopIteration:
            break
        nfr += 1
        for ix in range(nx):
            colUx, colUz = uvx[ix], uvz[ix]
            colVx, colVz = vvx[ix], vvz[ix]
            col = img[ix]
            if subtract_ref:
                colRx, colRz = rvx[ix], rvz[ix]
                for iz in range(nz):
                    ux = colUx[iz]-colRx[iz]; uz = colUz[iz]-colRz[iz]
                    col[iz] += scale*(ux*colVx[iz] + uz*colVz[iz])
            else:
                for iz in range(nz):
                    col[iz] += scale*(colUx[iz]*colVx[iz] + colUz[iz]*colVz[iz])
        if nfr % 50 == 0: print('frame', nfr, 'accumulated')
    print('%d frames accumulated' % nfr)
    return img

if __name__ == '__main__':
    nx, nz = int(sys.argv[1]), int(sys.argv[2])
    dx, dz = float(sys.argv[3]), float(sys.argv[4])
    lin_dir, nl_recv_dir, out = sys.argv[5], sys.argv[6], sys.argv[7]
    rho0, cp0 = float(sys.argv[8]), float(sys.argv[9])
    scale = 2.0/(rho0*cp0)

    def fn(d, prefix, field): return os.path.join(d, '%s_s%s.su'%(prefix, field))

    print('--- linear term: snap_ref x snap_recon_trev (REFSUB=1) ---')
    img_lin = crosscorrelate(
        fn(lin_dir,'snap_ref','vx'), fn(lin_dir,'snap_ref','vz'),
        fn(lin_dir,'snap_recon_trev','vx'), fn(lin_dir,'snap_recon_trev','vz'),
        nx, nz, scale)
    write_su(out+'_lin.su', nx, nz, img_lin); png(out+'_lin.png', nx, nz, img_lin)

    print('--- nonlinear term: (snap_forw-snap_ref) x snap_recon_trev (REFSUB=0, total data) ---')
    img_nl = crosscorrelate(
        fn(lin_dir,'snap_forw','vx'), fn(lin_dir,'snap_forw','vz'),
        fn(nl_recv_dir,'snap_recon_trev','vx'), fn(nl_recv_dir,'snap_recon_trev','vz'),
        nx, nz, scale, subtract_ref=(fn(lin_dir,'snap_ref','vx'), fn(lin_dir,'snap_ref','vz')))
    write_su(out+'_nl.su', nx, nz, img_nl); png(out+'_nl.png', nx, nz, img_nl)

    img_full = [[img_lin[ix][iz]+img_nl[ix][iz] for iz in range(nz)] for ix in range(nx)]
    write_su(out+'_full.su', nx, nz, img_full); png(out+'_full.png', nx, nz, img_full)
    print('done: %s_{lin,nl,full}.su/.png' % out)
