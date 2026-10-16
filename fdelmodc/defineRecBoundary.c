#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"bndsrc.h" /* pulls in fdelmodc.h; fdelmodc.h has no include guard, don't include it directly here too */

#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

void verr(char *fmt, ...);
void vmess(char *fmt, ...);

/*********************************************************************
*
* defineRecBoundary
*
* Builds the closed rectangular contour points+normals for [x1,x2] x
* [z1,z2], then builds one bndInterpPar table per staggered field
* (vx,vz,txx,tzz,txz) against that same contour, accounting for both
* the per-field boundary-padding index offset (mod.ioXx/ioXz, etc.)
* and the field's physical half-cell staggering offset (0 or dx/2,
* dz/2), so that field[ix*naz+iz] as used by the FD kernels is
* addressed correctly. See bndinterp.h/bndsrc.h for the recording vs.
* injection usage.
*
**********************************************************************/

int defineRecBoundary(modPar mod, float x1, float x2, float z1, float z2, bndSrcPar *bs, int verbose)
{
	int nptx, nptz, nbot, i, idx;
	float dx, dz, ox, oz;

	dx = mod.dx;
	dz = mod.dz;

	if (x2<=x1 || z2<=z1) verr("defineRecBoundary: rectangle [%f,%f]x[%f,%f] is degenerate", x1, x2, z1, z2);

	nptx = NINT((x2-x1)/dx)+1;
	nptz = NINT((z2-z1)/dz)+1-(mod.wfl_corner2 ? 0 : 2); /* default: interior points only, corners belong to top/bottom;
	                                                       * [CORNER2]: left/right faces also run over the corners */

	if (mod.wfl_open) nptz = 0; /* [OPEN] top face only */
	if (nptx<2 || nptz<0) verr("defineRecBoundary: rectangle too small for grid spacing dx=%f dz=%f", dx, dz);

	nbot = mod.wfl_open ? 0 : nptx; /* [OPEN] no bottom face */
	bs->n = nptx + nbot + 2*nptz;
	bs->xr = (float *)malloc(bs->n*sizeof(float));
	bs->zr = (float *)malloc(bs->n*sizeof(float));
	bs->nx = (float *)malloc(bs->n*sizeof(float));
	bs->nz = (float *)malloc(bs->n*sizeof(float));
	bs->is_corner = (int *)calloc(bs->n,sizeof(int));
	bs->edge_sign = mod.wfl_edgesign; /* [EDGESIGN] */
	bs->taper = (float *)malloc(bs->n*sizeof(float)); /* [TAPER] */
	for (i=0; i<bs->n; i++) bs->taper[i] = 1.0f;

	idx = 0;
	for (i=0; i<nptx; i++) { /* top face, outward normal points to smaller z (out of Omega) */
		bs->xr[idx] = x1+i*dx;
		bs->zr[idx] = z1;
		bs->nx[idx] = 0.0;
		bs->nz[idx] = -1.0;
		idx++;
	}
	for (i=0; i<nbot; i++) { /* bottom face */
		bs->xr[idx] = x1+i*dx;
		bs->zr[idx] = z2;
		bs->nx[idx] = 0.0;
		bs->nz[idx] = 1.0;
		idx++;
	}
	for (i=0; i<nptz; i++) { /* left face, excludes corners */
		bs->xr[idx] = x1;
		bs->zr[idx] = z1+(i+(mod.wfl_corner2 ? 0 : 1))*dz;
		bs->nx[idx] = -1.0;
		bs->nz[idx] = 0.0;
		idx++;
	}
	for (i=0; i<nptz; i++) { /* right face, excludes corners */
		bs->xr[idx] = x2;
		bs->zr[idx] = z1+(i+(mod.wfl_corner2 ? 0 : 1))*dz;
		bs->nx[idx] = 1.0;
		bs->nz[idx] = 0.0;
		idx++;
	}

	if (mod.wfl_taper>0) { /* [TAPER] quintic smoothstep ramp on the horizontal face(s): 0 at the end point,
	                        * 1 after wfl_taper points. w(t)=6t^5-15t^4+10t^3 (t=j/N) has zero VALUE, FIRST
	                        * *and* SECOND derivative at both t=0 and t=1, so it joins the flat w=1 interior
	                        * with no curvature discontinuity either (unlike the earlier raised-cosine
	                        * 0.5*(1-cos(pi*t)), which is only C1: its curvature jumps from -0.5*(pi/N)^2 to 0
	                        * at t=1). That curvature jump was still radiating a (smaller, but nonzero) amount
	                        * of edge-diffraction energy from each line end; the smoothstep further reduces it. */
		int nt2=mod.wfl_taper, j;
		if (2*nt2>nptx-1) verr("defineRecBoundary: wfl_taper=%d is too long for a face of %d points", nt2, nptx);
		for (i=0; i<nptx; i++) {
			j = (i<nptx-1-i) ? i : nptx-1-i;
			if (j<nt2) {
				float t = (float)j/(float)nt2;
				bs->taper[i] = t*t*t*(t*(t*6.0f-15.0f)+10.0f);
				if (nbot) bs->taper[nptx+i] = bs->taper[i]; /* bottom face has the same x points */
			}
		}
	}

	/* the 4 rectangle corners, always the first/last point of the top and
	 * bottom face loops given the construction above -- see injectBoundary.c
	 * for the weighting-factor-1/2 fix (Aaker et al. 2020, Sec. 5.1) */
	bs->is_corner[0]       = 1; /* (x1,z1), top face start */
	bs->is_corner[nptx-1]  = 1; /* (x2,z1), top face end */
	if (!mod.wfl_open) { /* [OPEN] for the open line the same two end points get weight 1/2 (trapezoid rule) */
		bs->is_corner[nptx]    = 1; /* (x1,z2), bottom face start */
		bs->is_corner[2*nptx-1]= 1; /* (x2,z2), bottom face end */
	}
	if (mod.wfl_corner2 && nptz>0) { /* [CORNER2] each corner is repeated on the vertical faces with the
	                                  * horizontal-normal, weight 1/2 (injectBoundary halves ds for flagged points):
	                                  * standard trapezoid rule on the closed contour */
		bs->is_corner[2*nptx]           = 1; /* left face,  (x1,z1) */
		bs->is_corner[2*nptx+nptz-1]    = 1; /* left face,  (x1,z2) */
		bs->is_corner[2*nptx+nptz]      = 1; /* right face, (x2,z1) */
		bs->is_corner[2*nptx+2*nptz-1]  = 1; /* right face, (x2,z2) */
	}

	if (verbose) {
		if (mod.wfl_open)
			vmess("Reconstruction boundary: OPEN line z=%f, x=[%f,%f], %d contour points", z1, x1, x2, bs->n);
		else
			vmess("Reconstruction boundary: [%f,%f] x [%f,%f], %d contour points (%d top/bottom, %d left/right)",
				x1, x2, z1, z2, bs->n, 2*nptx, 2*nptz);
	}

	/* Half-cell offsets: physical_x(field,idx) = idx*dx + ox_field. Derived
	 * from readModel.c's own construction (rox[(ix+ioXx)] is built from
	 * density AVERAGED between nominal cells ix and ix+1 -- i.e. it sits at
	 * nominal position (ix+0.5)*dx -- while l2m[(ix+ioPx)] uses density AT
	 * nominal cell ix directly, i.e. position ix*dx exactly), keeping every
	 * io* term explicit:
	 *   physical_x(l2m,idx) = (idx-ioPx)*dx
	 *   physical_x(rox,idx) = (idx-ioXx+0.5)*dx  =>  ox_vx = (0.5-ioXx)*dx
	 * Cross-checked against the elastic4.c stencil centering condition
	 * (txx[ix]-txx[ix-1] must center on vx[ix]'s own location), which
	 * requires ioXx-ioPx=1 -- true per getParameters.c (iorder/2 vs
	 * iorder/2-1) -- and reduces to the same ox_vx formula. Dropping the
	 * ioPx term while reasoning informally from the ASCII-art diagram
	 * gives the WRONG sign; both derivations must keep every io* term. */

	/* vx: half cell in x only */
	ox = 0.5*dx - mod.ioXx*dx;
	oz =        - mod.ioXz*dz;
	bs->vx = bndinterpInit(bs->xr, bs->zr, bs->n, ox, oz, dx, dz,
		mod.ioXx, mod.ieXx, mod.ioXz, mod.ieXz);

	/* vz: half cell in z only */
	ox =        - mod.ioZx*dx;
	oz = 0.5*dz - mod.ioZz*dz;
	bs->vz = bndinterpInit(bs->xr, bs->zr, bs->n, ox, oz, dx, dz,
		mod.ioZx, mod.ieZx, mod.ioZz, mod.ieZz);

	/* txx, tzz: collocated with the reference (P) grid, no half-cell shift */
	ox = -mod.ioPx*dx;
	oz = -mod.ioPz*dz;
	bs->txx = bndinterpInit(bs->xr, bs->zr, bs->n, ox, oz, dx, dz,
		mod.ioPx, mod.iePx, mod.ioPz, mod.iePz);
	bs->tzz = bndinterpInit(bs->xr, bs->zr, bs->n, ox, oz, dx, dz,
		mod.ioPx, mod.iePx, mod.ioPz, mod.iePz);

	/* txz: half cell in both x and z */
	ox = 0.5*dx - mod.ioTx*dx;
	oz = 0.5*dz - mod.ioTz*dz;
	bs->txz = bndinterpInit(bs->xr, bs->zr, bs->n, ox, oz, dx, dz,
		mod.ioTx, mod.ieTx, mod.ioTz, mod.ieTz);

	return 0;
}

void freeRecBoundary(bndSrcPar *bs)
{
	free(bs->xr);
	free(bs->zr);
	free(bs->nx);
	free(bs->nz);
	free(bs->is_corner);
	free(bs->taper);
	bndinterpFree(bs->vx);
	bndinterpFree(bs->vz);
	bndinterpFree(bs->txx);
	bndinterpFree(bs->tzz);
	bndinterpFree(bs->txz);
}
