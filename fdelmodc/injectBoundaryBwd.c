#include<stdlib.h>
#include<math.h>
#include"bndsrc.h"

/*********************************************************************
*
* injectBoundaryBwd   [BWD-KERNEL, opt-in via wfl_bwd=1]
*
* Identical to injectBoundary.c except that the medium coefficients
* sampled at the contour are un-negated (x -1): the caller has negated
* rox,roz,l2m,lam,mul (backward kernel, dt->-dt), and the sources must
* be built from |coef| with UNNEGATED reversed data -- see bndsrc.h.
* injectBoundary.c itself is untouched.
*
* Original header follows.
*
* See bndsrc.h for the full derivation. Summary:
*
*   force term (-> vx,vz):      d = rox_r * (ds_r/dz) * T(r)
*   deformation-rate (-> txx,tzz,txz): d = (ds_r/dz) * (stiffness-contracted v.n)
*
* The ds_r/dz factor (not dx*ds_r) is the standard FD point-source
* normalization: each boundary quadrature point represents a
* (density)*ds_r line-integral contribution, injected as a point
* source at a grid node of area dx*dz -- undoing rox/l2m/etc.'s own
* 1/dx (dx*ds_r) then dividing by the cell area dx*dz collapses to a
* net ds_r/dz. Cross-checked against applySource.c's existing point
* source convention (which carries the same net 1/(dx*dz) via its own
* explicit 1/dx pre-scale plus rox/l2m's built-in 1/dx) and by units
* (rox*dx*ds*tx has units m^3/s, not m/s; rox*(ds/dz)*tx has units
* m/s).
*
* where T(r) = traction = tau.n at contour point r (built from the
* recorded txx,tzz,txz and the point's outward normal), ds_r is the
* contour arc-length spacing at r (dx on top/bottom faces, dz on
* left/right faces), and rox_r/roz_r/l2m_r/lam_r/mul_r are the medium
* parameters sampled AT the contour point through that field's own
* bndInterpPar table (the same tables used for recording).
*
**********************************************************************/

int injectBoundaryBwd(bndSrcPar rb, int nt, int isam,
	float *vx, float *vz, float *txx, float *tzz, float *txz, int naz,
	float *rox, float *roz, float *l2m, float *lam, float *mul,
	float dx, float dz, int tshift, int vectorial,
	float *bs_vx, float *bs_vz, float *bs_txx, float *bs_tzz, float *bs_txz)
{
	float *rox_r, *roz_r, *l2m_r, *lam_r, *mul_r;
	float *d_vx, *d_vz, *d_txx, *d_tzz, *d_txz;
	int i, js;
	float tx, tz, vxr, vzr, nx, nz, ds;

	if (!rb.n) return 0;

	rox_r = (float *)malloc(rb.n*sizeof(float));
	roz_r = (float *)malloc(rb.n*sizeof(float));
	l2m_r = (float *)malloc(rb.n*sizeof(float));
	lam_r = (float *)malloc(rb.n*sizeof(float));
	mul_r = (float *)malloc(rb.n*sizeof(float));

	/* sample the medium at the contour points; cheap relative to a
	 * timestep and keeps this routine stateless (could be cached in
	 * bndSrcPar at defineRecBoundary time if this ever shows up in a
	 * profile) */
	bndinterp(0, 0, rox, naz, rox_r, rb.vx);
	bndinterp(0, 0, roz, naz, roz_r, rb.vz);
	bndinterp(0, 0, l2m, naz, l2m_r, rb.txx);
	bndinterp(0, 0, lam, naz, lam_r, rb.txx);
	bndinterp(0, 0, mul, naz, mul_r, rb.txz);

	for (i=0; i<rb.n; i++) { /* [BWD-KERNEL] arrays were negated by the caller: undo exactly (not fabsf; lam may be <0) */
		rox_r[i]=-rox_r[i]; roz_r[i]=-roz_r[i];
		l2m_r[i]=-l2m_r[i]; lam_r[i]=-lam_r[i]; mul_r[i]=-mul_r[i];
	}

	d_vx  = (float *)malloc(rb.n*sizeof(float));
	d_vz  = (float *)malloc(rb.n*sizeof(float));
	d_txx = (float *)malloc(rb.n*sizeof(float));
	d_tzz = (float *)malloc(rb.n*sizeof(float));
	d_txz = (float *)malloc(rb.n*sizeof(float));

	for (i=0; i<rb.n; i++) {
		nx = rb.nx[i];
		nz = rb.nz[i];
		ds = (fabsf(nz)>0.5f) ? dx : dz; /* top/bottom face (nz=+-1) -> dx spacing, left/right (nx=+-1) -> dz spacing */
		if (rb.is_corner[i]) ds *= 0.5f; /* Aaker et al. 2020 Sec. 5.1: the boundary normal is
		                                  * discontinuous at Cartesian-aligned corners -- "the
		                                  * greatest source of errors present" in their tests --
		                                  * fixed with a weighting factor of 1/2 there */

		/* [TSHIFT] The force source (built from tau) is added to v AFTER the stress update of this
		 * step, so tau only sees it one step later than the kernel's own applySource() ordering
		 * (v-update, force, tau-update). Injecting the force for the NEXT reversed step now
		 * (data index isam-1) gives the same v and puts the tau response on time. tshift=0: unchanged. */
		vxr = bs_vx[i*nt+isam];
		vzr = bs_vz[i*nt+isam];

		if (vectorial) { /* [VECTORIAL] Ravasi & Curtis (2013) eq. 8: recorded vx,vz routed through the
		                  * FORCE channel directly (no traction, no dot-with-normal -- vxr,vzr are already
		                  * the x,z components of the recorded vector); nothing injected into the stresses. */
			d_vx[i] = rox_r[i]*(ds/dz)*vxr;
			d_vz[i] = roz_r[i]*(ds/dz)*vzr;
			d_txx[i] = 0.0f; d_tzz[i] = 0.0f; d_txz[i] = 0.0f;
		}
		else {
			js = isam - tshift;
			if (js<0 || js>=nt) { tx = 0.0f; tz = 0.0f; }
			else {
				tx = bs_txx[i*nt+js]*nx + bs_txz[i*nt+js]*nz; /* traction T = tau.n */
				tz = bs_txz[i*nt+js]*nx + bs_tzz[i*nt+js]*nz;
			}

			d_vx[i] = rox_r[i]*(ds/dz)*tx;
			d_vz[i] = roz_r[i]*(ds/dz)*tz;

			d_txx[i] = (ds/dz)*( l2m_r[i]*vxr*nx + lam_r[i]*vzr*nz );
			d_tzz[i] = (ds/dz)*( lam_r[i]*vxr*nx + l2m_r[i]*vzr*nz );
			d_txz[i] = (ds/dz)*mul_r[i]*( vxr*nz + vzr*nx );
		}

		/* [EDGESIGN] first/last point of a horizontal face (or of the open line): all five sources times edge_sign
		 * (-1 = reversed polarity); edge_sign=1 leaves everything unchanged */
		if (rb.is_corner[i] && nz!=0.0f && rb.edge_sign!=1.0f) {
			d_vx[i]*=rb.edge_sign; d_vz[i]*=rb.edge_sign;
			d_txx[i]*=rb.edge_sign; d_tzz[i]*=rb.edge_sign; d_txz[i]*=rb.edge_sign;
		}
		/* [TAPER] smooth amplitude ramp at the ends of the horizontal face(s); 1 everywhere when wfl_taper=0 */
		if (rb.taper[i]!=1.0f) {
			d_vx[i]*=rb.taper[i]; d_vz[i]*=rb.taper[i];
			d_txx[i]*=rb.taper[i]; d_tzz[i]*=rb.taper[i]; d_txz[i]*=rb.taper[i];
		}
	}

	bndinterp(1, 0, vx,  naz, d_vx,  rb.vx);
	bndinterp(1, 0, vz,  naz, d_vz,  rb.vz);
	bndinterp(1, 0, txx, naz, d_txx, rb.txx);
	bndinterp(1, 0, tzz, naz, d_tzz, rb.tzz);
	bndinterp(1, 0, txz, naz, d_txz, rb.txz);

	free(rox_r); free(roz_r); free(l2m_r); free(lam_r); free(mul_r);
	free(d_vx); free(d_vz); free(d_txx); free(d_tzz); free(d_txz);

	return 0;
}
