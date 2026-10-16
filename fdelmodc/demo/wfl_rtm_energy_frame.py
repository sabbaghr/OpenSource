#!/usr/bin/env python3
"""
wfl_rtm_energy_frame.py -- SAME energy-norm imaging condition as wfl_rtm_energy.py, but for a single
snapshot frame k (direct seek, no need to sequentially read through the whole file), for building a
per-frame "how did the image build up over time" figure. Writes out.png (+ out.su).

  wfl_rtm_energy_frame.py sizex sizez dx dz case_dir out src_prefix frame
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wfl_rtm_energy import read_model, write_su, png
import array

def load_frame_all(fn, frame, nx, nz):
    tr = 240 + 4*nz
    with open(fn, 'rb') as f:
        f.seek(frame*nx*tr)
        b = f.read(nx*tr)
    cols = []
    for ix in range(nx):
        a = array.array('f'); a.frombytes(b[ix*tr+240:(ix+1)*tr]); cols.append(a)
    return cols

if __name__ == '__main__':
    sizex, sizez, dx, dz = [float(v) for v in sys.argv[1:5]]
    case_dir, out, src, frame = sys.argv[5], sys.argv[6], sys.argv[7], int(sys.argv[8])
    nx = int(round(sizex/dx))+1; nz = int(round(sizez/dz))+1

    cp = read_model(os.path.join(case_dir,'modelref_cp.su'), nx, nz)
    cs = read_model(os.path.join(case_dir,'modelref_cs.su'), nx, nz)
    ro = read_model(os.path.join(case_dir,'modelref_ro.su'), nx, nz)
    lam = [[ro[ix][iz]*(cp[ix][iz]**2 - 2*cs[ix][iz]**2) for iz in range(nz)] for ix in range(nx)]
    mu  = [[ro[ix][iz]*cs[ix][iz]**2 for iz in range(nz)] for ix in range(nx)]
    mu_max = max(max(row) for row in mu)
    mu_thresh = mu_max * 1e-6

    fields = ['vx','vz','txx','tzz','txz']
    S = {f: load_frame_all(os.path.join(case_dir,'snap_%s_s%s.su'%(src,f)), frame, nx, nz) for f in fields}
    R = {f: load_frame_all(os.path.join(case_dir,'snap_recon_trev_s%s.su'%f), frame, nx, nz) for f in fields}

    img = [[0.0]*nz for _ in range(nx)]
    for ix in range(nx):
        vxs,vxr = S['vx'][ix], R['vx'][ix]
        vzs,vzr = S['vz'][ix], R['vz'][ix]
        txxs,txxr = S['txx'][ix], R['txx'][ix]
        tzzs,tzzr = S['tzz'][ix], R['tzz'][ix]
        txzs,txzr = S['txz'][ix], R['txz'][ix]
        im = img[ix]
        for iz in range(nz):
            kinetic = ro[ix][iz]*(vxs[iz]*vxr[iz] + vzs[iz]*vzr[iz])
            Ss = txxs[iz]+tzzs[iz]; Sr = txxr[iz]+tzzr[iz]
            isotropic = Ss*Sr/(4.0*(lam[ix][iz]+mu[ix][iz]))
            m = mu[ix][iz]
            if m > mu_thresh:
                Ds = txxs[iz]-tzzs[iz]; Dr = txxr[iz]-tzzr[iz]
                deviatoric = Ds*Dr/(4.0*m) + txzs[iz]*txzr[iz]/m
            else:
                deviatoric = 0.0
            im[iz] = kinetic + isotropic + deviatoric
    write_su(out+'.su', nx, nz, img)
    png(out+'.png', nx, nz, img)
