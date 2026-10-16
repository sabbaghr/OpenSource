#!/usr/bin/env python3
"""
wfl_rtm_nonlinear_full.py -- full 3-term nonlinear source-receiver imaging condition
(Ravasi & Curtis 2013, eq. 21: 0S + S0 + SS), velocity-only crosscorrelation (no energy-norm
elastic-parameter weighting, per eq. 1's I_ji = G^{S(v,f)} form).

  v0  = snap_ref                                  (source-side reference, G^0)
  vS  = snap_forw - snap_ref                       (source-side scattered, G^S)
  vt0 = snap_recon_trev in dir_direct               (receiver-side reference, G-tilde^0:
        direct-arrival-only data back-propagated through the background model)
  vtS = snap_recon_trev in dir_sharp - snap_recon_trev in dir_total
        (receiver-side scattered, G-tilde^S: scattered data back-propagated through the exact
        model minus the same data back-propagated through the background model)

  I_0S = sum_t v0.vtS         I_S0 = sum_t vS.vt0         I_SS = sum_t vS.vtS
  I_linear = I_0S + I_S0      I_nonlinear = I_SS          I_full = I_linear + I_nonlinear

  wfl_rtm_nonlinear_full.py nx nz dx dz dir_sharp dir_direct dir_total out_prefix
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
    dir_sharp, dir_direct, dir_total, out = sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]

    def fn(d, prefix, field): return os.path.join(d, '%s_s%s.su'%(prefix, field))

    g_v0x = frames(fn(dir_sharp,'snap_ref','vx'), nx, nz)
    g_v0z = frames(fn(dir_sharp,'snap_ref','vz'), nx, nz)
    g_vFx = frames(fn(dir_sharp,'snap_forw','vx'), nx, nz)   # full source-side field (exact model)
    g_vFz = frames(fn(dir_sharp,'snap_forw','vz'), nx, nz)
    g_vt0x = frames(fn(dir_direct,'snap_recon_trev','vx'), nx, nz)
    g_vt0z = frames(fn(dir_direct,'snap_recon_trev','vz'), nx, nz)
    g_vtEx = frames(fn(dir_sharp,'snap_recon_trev','vx'), nx, nz)   # receiver-side, exact-model half
    g_vtEz = frames(fn(dir_sharp,'snap_recon_trev','vz'), nx, nz)
    g_vtTx = frames(fn(dir_total,'snap_recon_trev','vx'), nx, nz)   # receiver-side, background-model half
    g_vtTz = frames(fn(dir_total,'snap_recon_trev','vz'), nx, nz)

    img_0S = [[0.0]*nz for _ in range(nx)]
    img_S0 = [[0.0]*nz for _ in range(nx)]
    img_SS = [[0.0]*nz for _ in range(nx)]
    nfr = 0
    while True:
        try:
            v0x=next(g_v0x); v0z=next(g_v0z)
            vFx=next(g_vFx); vFz=next(g_vFz)
            vt0x=next(g_vt0x); vt0z=next(g_vt0z)
            vtEx=next(g_vtEx); vtEz=next(g_vtEz)
            vtTx=next(g_vtTx); vtTz=next(g_vtTz)
        except StopIteration:
            break
        nfr += 1
        for ix in range(nx):
            c_v0x, c_v0z = v0x[ix], v0z[ix]
            c_vSx = [vFx[ix][iz]-v0x[ix][iz] for iz in range(nz)]
            c_vSz = [vFz[ix][iz]-v0z[ix][iz] for iz in range(nz)]
            c_vt0x, c_vt0z = vt0x[ix], vt0z[ix]
            c_vtSx = [vtEx[ix][iz]-vtTx[ix][iz] for iz in range(nz)]
            c_vtSz = [vtEz[ix][iz]-vtTz[ix][iz] for iz in range(nz)]
            col0S, colS0, colSS = img_0S[ix], img_S0[ix], img_SS[ix]
            for iz in range(nz):
                col0S[iz] += c_v0x[iz]*c_vtSx[iz] + c_v0z[iz]*c_vtSz[iz]
                colS0[iz] += c_vSx[iz]*c_vt0x[iz] + c_vSz[iz]*c_vt0z[iz]
                colSS[iz] += c_vSx[iz]*c_vtSx[iz] + c_vSz[iz]*c_vtSz[iz]
        if nfr % 50 == 0: print('frame', nfr, 'accumulated')
    print('%d frames accumulated' % nfr)

    img_lin = [[img_0S[ix][iz]+img_S0[ix][iz] for iz in range(nz)] for ix in range(nx)]
    img_full = [[img_lin[ix][iz]+img_SS[ix][iz] for iz in range(nz)] for ix in range(nx)]

    write_su(out+'_0S.su', nx, nz, img_0S); png(out+'_0S.png', nx, nz, img_0S)
    write_su(out+'_S0.su', nx, nz, img_S0); png(out+'_S0.png', nx, nz, img_S0)
    write_su(out+'_lin.su', nx, nz, img_lin); png(out+'_lin.png', nx, nz, img_lin)
    write_su(out+'_SS.su', nx, nz, img_SS); png(out+'_SS.png', nx, nz, img_SS)
    write_su(out+'_full.su', nx, nz, img_full); png(out+'_full.png', nx, nz, img_full)
    print('done')
