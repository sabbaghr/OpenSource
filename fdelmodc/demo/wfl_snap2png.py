#!/usr/bin/env python3
"""
wfl_snap2png.py -- render frames of an fdelmodc snapshot file (SU, one trace per x column) to PNG (no dependencies).

  wfl_snap2png.py out.png nx nz clip frame file [file2 ...] [--box x1 x2 z1 z2 dx dz] [--diff]  [--scale s]

Draws the frames of the given files side by side (same colour scale +-clip: blue negative, white 0, red positive).
--diff : instead render file1 - file2 for a single frame.  --box draws the reconstruction box/line (model coords).
"""
import sys, struct, zlib, array
def load_frame(fn,frame,nx,nz):
    tr=240+4*nz; cols=[]
    with open(fn,'rb') as f:
        f.seek(frame*nx*tr)
        b=f.read(nx*tr)
    for ix in range(nx):
        a=array.array('f'); a.frombytes(b[ix*tr+240:(ix+1)*tr]); cols.append(a)
    return cols   # cols[ix][iz]
def color(v,clip,gray=False):
    t=max(-1.0,min(1.0,v/clip))
    if gray:
        g=int(255*(t+1)/2)  # -clip->black(0), 0->mid-grey(127), +clip->white(255)
        return (g,g,g)
    if t>=0: return (255,int(255*(1-t)),int(255*(1-t)))
    return (int(255*(1+t)),int(255*(1+t)),255)
def png(fn,w,h,rows):
    raw=b''.join(b'\x00'+bytes(r) for r in rows)
    def chunk(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    open(fn,'wb').write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
if __name__=='__main__':
    a=sys.argv[1:]; out=a[0]; nx=int(a[1]); nz=int(a[2]); clip=float(a[3]); frame=int(a[4])
    box=None; diff=False; scale=2; crop=None; zoom=1; gray=False
    files=[]; i=5
    while i<len(a):
        if a[i]=='--box': box=[float(v) for v in a[i+1:i+7]]; i+=7
        elif a[i]=='--diff': diff=True; i+=1
        elif a[i]=='--scale': scale=int(a[i+1]); i+=2
        elif a[i]=='--crop': crop=[int(float(v)) for v in a[i+1:i+5]]; i+=5   # cell indices ix0 ix1 iz0 iz1
        elif a[i]=='--zoom': zoom=int(a[i+1]); i+=2
        elif a[i]=='--gray': gray=True; i+=1
        else: files.append(a[i]); i+=1
    if diff:
        A=load_frame(files[0],frame,nx,nz); B=load_frame(files[1],frame,nx,nz)
        panels=[[array.array('f',[x-y for x,y in zip(ca,cb)]) for ca,cb in zip(A,B)]]
    else: panels=[load_frame(f,frame,nx,nz) for f in files]
    if crop:
        ix0,ix1,iz0,iz1=crop
        panels=[[c[iz0:iz1] for c in cols[ix0:ix1]] for cols in panels]
        nx=ix1-ix0; nz=iz1-iz0; box=None
    gap=6; W=len(panels)*(nx//scale)+(len(panels)-1)*gap; H=nz//scale
    rows=[[255]*(3*W) for _ in range(H)]
    for p,cols in enumerate(panels):
        x0=p*(nx//scale+gap)
        for ix in range(0,nx-scale+1,scale):
            col=cols[ix]
            for iz in range(0,nz-scale+1,scale):
                # average over the scale x scale block for a smooth image
                v=0.0
                for dx_ in range(scale):
                    c2=cols[ix+dx_]
                    for dz_ in range(scale): v+=c2[iz+dz_]
                v/=scale*scale
                r,g,b=color(v,clip,gray); o=3*(x0+ix//scale)
                rows[iz//scale][o]=r; rows[iz//scale][o+1]=g; rows[iz//scale][o+2]=b
        if box:
            bx1,bx2,bz1,bz2,ddx,ddz=box
            X1=int(bx1/ddx)//scale; X2=int(bx2/ddx)//scale; Z1=int(bz1/ddz)//scale; Z2=int(bz2/ddz)//scale
            for xx in range(X1,X2+1):
                for zz in (Z1,Z2):
                    if 0<=zz<H: o=3*(x0+xx); rows[zz][o:o+3]=[0,150,0]
            for zz in range(Z1,Z2+1):
                for xx in (X1,X2):
                    if 0<=zz<H: o=3*(x0+xx); rows[zz][o:o+3]=[0,150,0]
    if zoom>1:
        rows=[sum(([px for px in [r[3*i+c] for c in range(3)]]*zoom for i in range(W)),[]) for r in rows for _ in range(zoom)]
        W*=zoom; H*=zoom
    png(out,W,H,rows)
    print('wrote',out,W,'x',H)
