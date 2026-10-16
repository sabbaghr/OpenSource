#!/usr/bin/env python3
"""
wfl_ablate_analyze.py -- outside-leakage metric for reconstruction snapshots.

usage: wfl_ablate_analyze.py sizex sizez dx dz x1 x2 z1 z2 field[,field..] prefix[:label] [prefix[:label] ...]

For every prefix, reads <prefix>_<field>.su (fields as named by fdelmodc, e.g. svx svz),
reverses nothing (the metric does not need the forward run): the reconstruction is
ideally ZERO outside Omega, so the rms of the reconstructed field in zones OUTSIDE Omega,
relative to the rms of the reconstructed field INSIDE Omega (same file), measures the leak.
Zones (cells 12..40 outside the box, i.e. 48..160 m at dx=4):
  top / bottom / sides : outside the central third of each face
  corner_top / corner_bot : outside cells within Chebyshev distance 12..40 of the two top / two bottom corners
This is independent of any time lag between forward and reconstruction.
"""
import sys, array, os, math
sizex,sizez,dx,dz,x1,x2,z1,z2=[float(v) for v in sys.argv[1:9]]
fields=sys.argv[9].split(',')
items=[a.split(':') if ':' in a else [a,a] for a in sys.argv[10:]]
nx=int(round(sizex/dx))+1; ns=int(round(sizez/dz))+1; tr=240+ns*4
ix1,ix2,iz1,iz2=[int(round(v)) for v in (x1/dx,x2/dx,z1/dz,z2/dz)]
mx=(ix1+ix2)//2; mz=(iz1+iz2)//2; tx=(ix2-ix1)//6; tz=(iz2-iz1)//6
A,B=12,40
OPEN=os.environ.get('WFL_OPEN','0')=='1'  # open (single top line) surface: only the zone above the line is meaningful
def inom(ix,iz): return ix1<=ix<=ix2 and iz1<=iz<=iz2
zones={'top':[], 'bottom':[], 'sides':[], 'corner_top':[], 'corner_bot':[], 'inside':[]}
for ix in range(nx):
    for iz in range(ns):
        if inom(ix,iz):
            if ix1+10<=ix<=ix2-10 and iz1+10<=iz<=iz2-10: zones['inside'].append((ix,iz))
            continue
        if abs(ix-mx)<=tx and A<=iz1-iz<=B: zones['top'].append((ix,iz))
        elif abs(ix-mx)<=tx and A<=iz-iz2<=B: zones['bottom'].append((ix,iz))
        elif abs(iz-mz)<=tz and (A<=ix1-ix<=B or A<=ix-ix2<=B): zones['sides'].append((ix,iz))
        elif any(A<=max(abs(ix-cx),abs(iz-cz))<=B for cx,cz in ((ix1,iz1),(ix2,iz1))): zones['corner_top'].append((ix,iz))
        elif any(A<=max(abs(ix-cx),abs(iz-cz))<=B for cx,cz in ((ix1,iz2),(ix2,iz2))): zones['corner_bot'].append((ix,iz))
def frames(fn):
    n=os.path.getsize(fn)//(tr*nx)
    with open(fn,'rb') as f:
        for k in range(n):
            b=f.read(tr*nx); cols=[]
            for ix in range(nx):
                a=array.array('f'); a.frombytes(b[ix*tr+240:(ix+1)*tr]); cols.append(a)
            yield cols
print("zones (cells): "+", ".join("%s=%d"%(k,len(v)) for k,v in zones.items()))
for fld in fields:
    print("\nfield %s: outside-zone rms / inside-Omega rms (same file)     [lower = less leak]%s"%(fld,'   (OPEN line: only the zones above the line apply)' if OPEN else ''))
    print("%-28s %8s %8s %8s %9s %9s   %s"%("variant","top","bottom","sides","corner_top","corner_bot","inside rms"))
    for prefix,label in items:
        fn='%s_%s.su'%(prefix,fld)
        if not os.path.exists(fn): print("%-28s missing %s"%(label,fn)); continue
        ss={k:0.0 for k in zones}; nfr=0
        for cols in frames(fn):
            nfr+=1
            for z,cells in zones.items():
                s=0.0
                for ix,iz in cells:
                    v=cols[ix][iz]; s+=v*v
                ss[z]+=s
        rms={z:(math.sqrt(ss[z]/(nfr*len(zones[z]))) if zones[z] else 0.0) for z in zones}  # a zone can be empty (box/line reaching the model edge)
        if OPEN:  # only the zone above the line is outside the reconstruction region
            print("%-28s %8.4f %8s %8s %9s %9s   %.3e"%(label,rms['top']/rms['inside'],'n/a','n/a','n/a','n/a',rms['inside']))
        else:
            print("%-28s %8.4f %8.4f %8.4f %9.4f %9.4f   %.3e"%(label,rms['top']/rms['inside'],rms['bottom']/rms['inside'],rms['sides']/rms['inside'],rms['corner_top']/rms['inside'],rms['corner_bot']/rms['inside'],rms['inside']))
