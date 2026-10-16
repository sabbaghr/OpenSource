#ifndef _bndsrc_h
#define _bndsrc_h

#include"fdelmodc.h"
#include"bndinterp.h"

/*********************************************************************
*
* Closed rectangular reconstruction boundary (Mittet 1994 / Aaker et
* al. 2020, GJI 222, eqs. 23-24): a one-point-thin contour enclosing
* the region Omega to be reconstructed, plus one bndInterpPar table
* per staggered field (vx, vz, txx, tzz, txz), all built from the
* SAME contour points (xr,zr) and outward normals (nx,nz).
*
* Because each field's table is built once and then reused with
* inject=0 (record) or inject=1 (inject) via bndinterp(), recording
* and reconstruction-injection are guaranteed to be exact adjoints of
* each other -- see bndinterp.h.
*
*********************************************************************/

typedef struct _bndSrcPar {
	int n;             /* number of contour points */
	float *xr, *zr;    /* contour point position, relative to mod.x0/mod.z0 (same convention as rec->xr/zr) */
	float *nx, *nz;    /* outward unit normal at each contour point */
	int *is_corner;    /* 1 at the 4 rectangle corners, 0 elsewhere -- see injectBoundary.c: Aaker et
	                     * al. 2020 Sec. 5.1 identify the discontinuous normal at Cartesian-aligned
	                     * corners as "the greatest source of errors present" and fix it with a
	                     * weighting factor of 1/2 on the corner points themselves. */
	bndInterpPar *vx, *vz, *txx, *tzz, *txz;
	float *taper;      /* [TAPER] per-contour-point amplitude factor (1 everywhere unless wfl_taper>0) */
	float edge_sign;   /* [EDGESIGN] factor on all five sources at the end points of the horizontal face(s); 1 = unchanged */
} bndSrcPar;

/*
 * x1,x2,z1,z2: rectangle bounds of Omega, in the same real-coordinate
 *              convention as rec->xr/zr (relative to mod.x0/mod.z0).
 *              Contour points are sampled at spacing dx along the top
 *              /bottom faces and dz along the left/right faces; the 4
 *              corners belong to the top/bottom faces only (avoids
 *              duplicate points at zero measure).
 */
int defineRecBoundary(modPar mod, float x1, float x2, float z1, float z2, bndSrcPar *bs, int verbose);

void freeRecBoundary(bndSrcPar *bs);

/*
 * Recording pass: extracts vx,vz,txx,tzz,txz at all rb.n contour
 * points (bndinterp(inject=0,...) on each field's table) and stores
 * them at sample isam, in the same (point*nt+isam) layout as
 * getRecTimes' rec_* arrays. bs_vx..bs_txz must each be preallocated
 * to rb.n*nt floats. naz is the field arrays' z-stride (mod.naz).
 */
int storeBoundaryWav(bndSrcPar rb, int nt, int isam,
	float *vx, float *vz, float *txx, float *tzz, float *txz, int naz,
	float *bs_vx, float *bs_vz, float *bs_txx, float *bs_tzz, float *bs_txz);

/*
 * Reconstruction pass: injects the modified source functions of Aaker
 * et al. (2020, GJI 222, eqs. 23-24) at time sample isam, using the
 * SAME bndinterp tables (with inject=1) that storeBoundaryWav used
 * (with inject=0) to record bs_vx..bs_txz -- see bndinterp.h.
 *
 * Physics scaling is derived directly from elastic4.c's discretized
 * update equations (not from the paper's continuous eqs. taken in
 * isolation): rox=dt/(dx*rho) etc. already carry the dt/rho and 1/dx
 * factors elastic4.c's own stencils rely on, and each boundary point
 * represents a (density)*ds line-integral contribution (arc-length
 * quadrature weight ds: dx on top/bottom faces, dz on left/right,
 * inferred from each point's normal). Injecting that contribution as
 * a point source at a grid node of area dx*dz (the standard FD
 * point-source normalization -- same one applySource.c's existing
 * sources use, via their own explicit 1/dx pre-scale plus the 1/dx
 * built into rox/l2m/etc.) works out to a net factor of ds/dz, not
 * dx*ds -- an earlier version of this code used dx*ds and was off by
 * a factor of dx*dz (verified by a units check: rox*dx*ds*T has units
 * m^3/s, not m/s; rox*(ds/dz)*T has units m/s). roz/l2m/lam/mul are
 * all built from fdelmodc's OWN fac=dt/dx (used uniformly for both
 * axes, readModel.c) rather than a separate dt/dz, so dz is the
 * uniform normalizer throughout, not a naive per-axis dx/dz split.
 *
 * The density ratio rho(x)/rho(x_r) in eq. 23 is approximated as 1
 * (rox/roz/l2m/lam/mul are sampled AT the contour point via that
 * field's own table, i.e. evaluated in the immediate neighbourhood of
 * both x and x_r, standard for the sub-cell distances involved).
 *
 * CAVEAT: this scaling has been checked for internal/dimensional
 * consistency against elastic4.c's stencils, not yet against an
 * end-to-end forward-model-then-reconstruct numerical test.
 */
/* tshift ([TSHIFT], wfl_tshift): traction (force-source) data taken from sample isam-tshift; 0 = original behaviour */
/* vectorial ([VECTORIAL], wfl_vectorial): Ravasi & Curtis (2013) eq. 8 heuristic -- vx,vz routed
 * through the force channel directly, nothing injected into txx,tzz,txz; 0 = original (tensorial) behaviour */
int injectBoundary(bndSrcPar rb, int nt, int isam,
	float *vx, float *vz, float *txx, float *tzz, float *txz, int naz,
	float *rox, float *roz, float *l2m, float *lam, float *mul,
	float dx, float dz, int tshift, int vectorial,
	float *bs_vx, float *bs_vz, float *bs_txx, float *bs_tzz, float *bs_txz);

/*
 * [BWD-KERNEL] Backward-kernel variant of injectBoundary (opt-in, wfl_bwd=1).
 * Used when the update kernel is run with dt->-dt (rox,roz,l2m,lam,mul all
 * negated after readModel). Samples the medium at the contour points and
 * un-negates it (x -1, i.e. the same |coef| as the forward-kernel version) and injects
 * UNNEGATED reversed data with sources (+,+) in terms of |coef|:
 *   d_vx = |rox|(ds/dz) T_x ,  d_txx = (ds/dz)( |l2m| vx nx + |lam| vz nz ) ...
 * so both v and sigma reconstruct with their TRUE sign (no Option S/V flip).
 * Same corner weighting, ds and index mapping as injectBoundary.
 */
int injectBoundaryBwd(bndSrcPar rb, int nt, int isam,
	float *vx, float *vz, float *txx, float *tzz, float *txz, int naz,
	float *rox, float *roz, float *l2m, float *lam, float *mul,
	float dx, float dz, int tshift, int vectorial,
	float *bs_vx, float *bs_vz, float *bs_txx, float *bs_tzz, float *bs_txz);

int writeBoundaryWav(char *file, bndSrcPar rb, int nt,
	float *bs_vx, float *bs_vz, float *bs_txx, float *bs_tzz, float *bs_txz);

int readBoundaryWav(char *file, bndSrcPar rb, int nt,
	float **bs_vx, float **bs_vz, float **bs_txx, float **bs_tzz, float **bs_txz);

#endif
