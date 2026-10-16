#!/usr/bin/env python3
"""
wfl_rtm_energy_frame_breakdown.py -- per-frame (not cumulative) breakdown of the modified
(attenuated, Rocha eq. 39) energy-norm imaging condition for the tensorial, sharp-migration-model
Rocha case. For each selected snapshot index, writes 3 panels:

  forw_fNNN  = source-side forward wavefield (snap_<src_prefix>, sharp model), vz component
  recon_fNNN = receiver-side reconstructed wavefield (snap_recon_trev), vz component
  img_fNNN   = that single frame's contribution to the ATTENUATED energy-norm image
               (-kinetic + potential, eq. 39 -- no time summation, just this one frame)

Each frame is read directly via seek (frames are fixed-size: nx*(240+4*nz) bytes), so this does not
need to scan the whole multi-hundred-MB snapshot file for each requested index.

  wfl_rtm_energy_frame_breakdown.py nx nz dx dz case_dir model_prefix src_prefix out_dir f1 f2 f3 ...
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

def read_frame(fn, nx, nz, f):
    tr = 240 + 4*nz
    frsize = tr*nx
    cols = []
    with open(fn, 'rb') as fh:
        fh.seek(f*frsize)
        b = fh.read(frsize)
    for ix in range(nx):
        a = array.array('f'); a.frombytes(b[ix*tr+240:(ix+1)*tr]); cols.append(a)
    return cols

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
    if clip == 0: clip = 1.0
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
    case_dir, model_prefix, src_prefix, out_dir = sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]
    frame_idx = [int(x) for x in sys.argv[9:]]
    os.makedirs(out_dir, exist_ok=True)

    cp = read_model(os.path.join(case_dir, model_prefix+'_cp.su'), nx, nz)
    cs = read_model(os.path.join(case_dir, model_prefix+'_cs.su'), nx, nz)
    ro = read_model(os.path.join(case_dir, model_prefix+'_ro.su'), nx, nz)
    mu  = [[ro[ix][iz]*cs[ix][iz]**2 for iz in range(nz)] for ix in range(nx)]
    lam = [[ro[ix][iz]*(cp[ix][iz]**2 - 2*cs[ix][iz]**2) for iz in range(nz)] for ix in range(nx)]

    def fn(prefix, field): return os.path.join(case_dir, '%s_s%s.su'%(prefix, field))

    for f in frame_idx:
        Uf = {field: read_frame(fn(src_prefix, field), nx, nz, f) for field in ['vx','vz','txx','tzz','txz']}
        Vf = {field: read_frame(fn('snap_recon_trev', field), nx, nz, f) for field in ['vx','vz','txx','tzz','txz']}

        img_atten = [[0.0]*nz for _ in range(nx)]
        for ix in range(nx):
            txxU, tzzU, txzU = Uf['txx'][ix], Uf['tzz'][ix], Uf['txz'][ix]
            txxV, tzzV, txzV = Vf['txx'][ix], Vf['tzz'][ix], Vf['txz'][ix]
            vxU, vzU = Uf['vx'][ix], Uf['vz'][ix]
            vxV, vzV = Vf['vx'][ix], Vf['vz'][ix]
            muc, lamc = mu[ix], lam[ix]
            acol = img_atten[ix]
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
                acol[iz] = -kinetic + potential

        write_su(os.path.join(out_dir, 'forw_f%03d.su'%f), nx, nz, Uf['vz'])
        png(os.path.join(out_dir, 'forw_f%03d.png'%f), nx, nz, Uf['vz'])
        write_su(os.path.join(out_dir, 'recon_f%03d.su'%f), nx, nz, Vf['vz'])
        png(os.path.join(out_dir, 'recon_f%03d.png'%f), nx, nz, Vf['vz'])
        write_su(os.path.join(out_dir, 'img_f%03d.su'%f), nx, nz, img_atten)
        png(os.path.join(out_dir, 'img_f%03d.png'%f), nx, nz, img_atten)
        print('frame %d done' % f)
    print('all frames done')
