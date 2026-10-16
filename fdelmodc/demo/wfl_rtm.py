#!/usr/bin/env python3
"""
wfl_rtm.py -- zero-lag cross-correlation RTM image from a forward run and a time-reversed
reconstruction produced by wfl_case.sh (snap_forw_s<c>.su and snap_recon_trev_s<c>.su, already
frame-aligned: frame k of _trev matches frame k of _forw).

  wfl_rtm.py sizex sizez dx dz case_dir out_prefix field[,field2,...] [--norm] [--src PREFIX]

--src PREFIX (default 'snap_forw'): the source-side wavefield file prefix, i.e. <PREFIX>_s<field>.su.
Use --src snap_ref to correlate against the REFERENCE/background-model forward wavefield (needed for a
proper linearized imaging condition when the receiver side only contains the scattered field, e.g. from
REFSUB/MUTE in wfl_case.sh) instead of the default snap_forw (the true/total-model forward field, which
is only appropriate when the receiver side ALSO contains the total field, uncommon for RTM).

--zmin Z --zmax Z (metres, default: whole model): restrict the correlation SUM to this depth range.
Real RTM never images above/at the receiver array for exactly the reason seen here: both wavefields are
large and share common (non-reflector) structure near the source/array, so leaving that band in the sum
produces a strong near-source artifact that swamps any real, weak, deep reflector signal.

Image (source wavefield S = forward, receiver wavefield R = reconstructed):
    I(x,z) = sum_t  sum_field  S_field(x,z,t) * R_field(x,z,t)
--norm additionally divides by the source illumination sum_t sum_field S_field(x,z,t)^2 (+ eps),
the standard way to balance amplitude with depth/offset.

Writes <out_prefix>.su (nx traces, one "frame", ns=nz -- a vertical section, viewable with
suximage/suxwigb) and, if --norm, <out_prefix>_norm.su. Also writes a quick-look PNG of each
(diverging colour scale, percentile clipped) via wfl_snap2png's colour function.
"""
import sys, struct, array, os, math

def read_geom(fn):
    d=open(fn,'rb').read(240); ns=struct.unpack('<H',d[114:116])[0]; return ns

def frames(fn, nx, nz):
    tr=240+4*nz
    n=os.path.getsize(fn)//(tr*nx)
    with open(fn,'rb') as f:
        for k in range(n):
            b=f.read(tr*nx)
            cols=[]
            for ix in range(nx):
                a=array.array('f'); a.frombytes(b[ix*tr+240:(ix+1)*tr]); cols.append(a)
            yield cols

def write_su(fn, nx, nz, img):
    tr=240+4*nz
    out=bytearray()
    for ix in range(nx):
        h=bytearray(240)
        struct.pack_into('<i',h,0,ix+1); struct.pack_into('<i',h,4,ix+1)
        struct.pack_into('<H',h,114,nz); struct.pack_into('<H',h,116,1)
        out+=h; out+=struct.pack('<%df'%nz,*img[ix])
    open(fn,'wb').write(out)

def png(fn, nx, nz, img, scale=2):
    def color(v,clip):
        t=max(-1.0,min(1.0,v/clip)) if clip else 0.0
        if t>=0: return (255,int(255*(1-t)),int(255*(1-t)))
        return (int(255*(1+t)),int(255*(1+t)),255)
    vals=sorted(abs(v) for col in img for v in col)
    clip=vals[int(0.98*len(vals))] if vals else 1.0
    W=nx//scale; H=nz//scale
    rows=[[255]*(3*W) for _ in range(H)]
    for ix in range(0,nx-scale+1,scale):
        col=img[ix]
        for iz in range(0,nz-scale+1,scale):
            v=sum(img[ix+dx_][iz+dz_] for dx_ in range(scale) for dz_ in range(scale))/(scale*scale)
            r,g,b=color(v,clip); o=3*(ix//scale)
            rows[iz//scale][o]=r; rows[iz//scale][o+1]=g; rows[iz//scale][o+2]=b
    raw=b''.join(b'\x00'+bytes(r) for r in rows)
    import zlib
    def chunk(t,dd): return struct.pack('>I',len(dd))+t+dd+struct.pack('>I',zlib.crc32(t+dd)&0xffffffff)
    open(fn,'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
    print('wrote',fn,W,'x',H,'clip=%.3e'%clip)

if __name__=='__main__':
    sizex,sizez,dx,dz=[float(v) for v in sys.argv[1:5]]
    case_dir=sys.argv[5]; out=sys.argv[6]; fields=sys.argv[7].split(',')
    rest=sys.argv[8:]
    norm='--norm' in rest
    src='snap_forw'
    if '--src' in rest: src=rest[rest.index('--src')+1]
    zmin=float(rest[rest.index('--zmin')+1]) if '--zmin' in rest else None
    zmax=float(rest[rest.index('--zmax')+1]) if '--zmax' in rest else None
    nx=int(round(sizex/dx))+1; nz=int(round(sizez/dz))+1
    iz0=int(round(zmin/dz)) if zmin is not None else 0
    iz1=int(round(zmax/dz))+1 if zmax is not None else nz
    img=[[0.0]*nz for _ in range(nx)]
    ill=[[0.0]*nz for _ in range(nx)]
    for f in fields:
        fwd_fn=os.path.join(case_dir,'%s_s%s.su'%(src,f))
        rec_fn=os.path.join(case_dir,'snap_recon_trev_s%s.su'%f)
        nF=read_geom(fwd_fn); nR=read_geom(rec_fn)
        assert nF==nz and nR==nz, (fwd_fn,rec_fn,nF,nR,nz)
        nfr=0
        for S,R in zip(frames(fwd_fn,nx,nz), frames(rec_fn,nx,nz)):
            nfr+=1
            for ix in range(nx):
                s=S[ix]; r=R[ix]; im=img[ix]; il=ill[ix]
                for iz in range(iz0,iz1):
                    im[iz]+=s[iz]*r[iz]; il[iz]+=s[iz]*s[iz]
        print('field %s: %d frames accumulated'%(f,nfr))
    write_su(out+'.su', nx, nz, img)
    png(out+'.png', nx, nz, img)
    if norm:
        eps=max(max(row) for row in ill)*1e-4 + 1e-300
        nimg=[[img[ix][iz]/(ill[ix][iz]+eps) for iz in range(nz)] for ix in range(nx)]
        write_su(out+'_norm.su', nx, nz, nimg)
        png(out+'_norm.png', nx, nz, nimg)
