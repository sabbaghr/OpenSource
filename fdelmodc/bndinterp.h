#ifndef _bndinterp_h
#define _bndinterp_h

/*********************************************************************
*
* Shared extract/inject primitive for closed-boundary wavefield
* reconstruction (Aaker et al., 2020, GJI 222).
*
* One weight table per staggered field (vx, vz, txx, tzz, txz), built
* once from each boundary point's real-valued (x,z) position against
* that field's own staggered-grid origin. The SAME table then drives
* both directions:
*
*   - inject=0 (extract): read the field at the boundary point during
*     the forward/recording pass.
*   - inject=1 (inject):  spread reconstruction-source data into the
*     field at the boundary point during the reconstruction pass.
*
* Because both directions reuse the identical weights (only the
* accumulation direction differs), injection is the exact adjoint of
* extraction by construction -- there is no separate hand-tuned
* stencil to keep in sync.
*
* NOTE: unlike a standalone linear-operator "model" vector, `field`
* here is the live, time-stepped wavefield array shared with the FD
* kernel. Injection is therefore always a pure accumulation (+=) --
* it must NEVER zero `field` first, unlike some LOP-style inject/
* extract operators that reset the destination when add=0.
*
*********************************************************************/

typedef struct _bndInterpPar {
	int n;                       /* number of boundary points        */
	int *ix, *iz;                /* base (floor) grid index per point */
	float *w00, *w01, *w10, *w11;/* bilinear weights per point        */
	int *valid;                  /* 1 if point has a full 2x2 footprint inside the field's usable grid range */
} bndInterpPar;

/*
 * xr, zr : real-valued (x,z) position of each boundary point, npt of them,
 *          in the same coordinate convention as rec->xr/rec->zr (offset
 *          from the model origin sub_x0/sub_z0).
 * ox, oz : the target field's own staggered-grid offset from that same
 *          origin (e.g. 0 for txx/tzz, dx/2 for vx, dz/2 for vz,
 *          dx/2+dz/2 for txz), such that field[ix*naz+iz] sits physically
 *          at (ix*dx+ox, iz*dz+oz).
 * ix0,ix1,iz0,iz1 : usable index range [ix0,ix1) x [iz0,iz1) for this
 *          field (e.g. mod.ioXx/mod.ieXx for vx) -- a point is valid only
 *          if its full 2x2 interpolation footprint lies inside this range.
 */
bndInterpPar *bndinterpInit(float *xr, float *zr, int npt,
	float ox, float oz, float dx, float dz,
	int ix0, int ix1, int iz0, int iz1);

void bndinterpFree(bndInterpPar *bi);

/*
 * inject=0: extract -- d[i] (= or +=, per add) <- bilinear sample of field at point i
 * inject=1: inject  -- field += d[i] * bilinear weights at point i (add is ignored)
 * naz     : z-stride of field, i.e. field[ix*naz+iz] (mod.naz)
 */
void bndinterp(int inject, int add, float *field, int naz, float *d, bndInterpPar *bi);

#endif
