#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"
#include"genfft.h"

/**
 * lag_modesep.c -- P/S MODE-SEPARATED time-lag gradients  (lag_modesep=1)
 *
 * WHAT
 * ----
 * The lag gradient (lag_gradient.c) is a sum of correlations between a
 * FORWARD factor and an ADJOINT factor:
 *
 *   g_lam      <-  conj(exx+ezz)^  (pxx+pzz)^
 *   g_mu       <-  2[conj(exx)^ pxx^ + conj(ezz)^ pzz^] + conj(shr)^ pxz^
 *   g_rho_dir  <-  conj(dsx)^ pvx^ + conj(dsz)^ pvz^
 *
 * (^ = the running DFTs; see lag_gradient.c for the names and grids).  Split
 * every factor into the part carried by the P wavefield and the part
 * carried by the S wavefield, F = F_P + F_S.  Each kernel is bilinear, so
 *
 *   g = g(PP) + g(PS) + g(SP) + g(SS)        (forward mode, adjoint mode)
 *
 * EXACTLY.  This file writes the PP and SS lag cubes of g_Vs and g_rho; the
 * cross terms are total - PP - SS.  "PP" is the gradient one gets by running
 * everything with only the P part of both wavefields, "SS" with only S.
 *
 * WHY g_Vp HAS NO FILE.  g_Vp = 2 rho Vp g_lam, and g_lam correlates the two
 * TRACES of strain.  An S wavefield is divergence free, so its strain is
 * traceless and g_lam(PS) = g_lam(SP) = g_lam(SS) = 0: g_Vp is PP, all of
 * it, by construction.  Every PP/SS difference shows up in g_Vs and g_rho.
 * The modesep.txt report checks this numerically instead of asserting it.
 *
 * HOW THE SPLIT IS MADE -- AN EXACT DISCRETE HELMHOLTZ PROJECTION
 * ---------------------------------------------------------------
 * The projection is linear and time invariant, so it commutes with the DFT:
 * splitting each accumulated FREQUENCY slice is the same as splitting the
 * wavefield at every time step and accumulating the parts.  It is done here,
 * after the time loop, nfreq times per factor -- the time loop and the
 * conventional cubes are untouched.
 *
 * (Implemented with bounded projector symbols, see msBuildSymbols; the
 * potential L^-1 theta itself is never formed.)
 * With the code's own staggered operators (D+ maps Vx->P along x, D- maps
 * P->Vx; same for z; coefficients c1..c4 of iorder), write G = (D-x, D-z) and
 * L = D+x D-x + D+z D-z.  On a periodic grid L has the REAL, even symbol
 *     L^(k) = -(4/dx^2) S(kx dx)^2 - (4/dz^2) S(kz dz)^2,
 *     S(k)  = c1 sin(k/2) + c2 sin(3k/2) + c3 sin(5k/2) + c4 sin(7k/2),
 * so the FFT sign convention cannot matter.  Then:
 *
 *  - VECTOR fields f = (fx on Vx, fz on Vz):
 *        f_P = G L^-1 (D+x fx + D+z fz),   f_S = f - f_P.
 *    div f_S = 0 and curl f_P = 0 hold to round-off.  Used for the density
 *    factors.  The forward factor dsx = -rox Dsig_vx is the velocity
 *    increment of one step (physical units), split as is.  The adjoint psi_v
 *    is a DUAL variable (psi_v = rho * physical velocity); it is mapped to
 *    the physical field by the buoyancy, (rox psi_vx, roz psi_vz), split, and
 *    mapped back.  rox and roz share the factor dt/dx, so it cancels.
 *
 *  - STRAIN fields e = (exx, ezz on P, shear on Txz) of a displacement w:
 *        w_P = G L^-1 (exx + ezz),  e_P = (D+x w_Px, D+z w_Pz,
 *                                          D-z w_Px + D-x w_Pz),
 *    so the P part of a strain needs only its TRACE.  The forward factors
 *    are the strain rate of v (lagGradientAccum's stencils).  The adjoint
 *    psi_sigma is ALSO an exact discrete strain: elastic4_adj.c updates it as
 *        txx += D+x(rox vx), tzz += D+z(roz vz), txz += D-z(rox vx)+D-x(roz vz),
 *    i.e. psi_sigma = strain of w = sum_t (rox psi_vx, roz psi_vz) on the same
 *    stencils.  No material parameter enters either split.
 *    tr(e_S) = 0 to round-off, which is why g_lam is pure PP (checked).
 *
 * WHAT IS NOT EXACT (read before interpreting the cubes)
 * -----------------------------------------------------
 *  - A Helmholtz split is not a MODE split where the medium varies fast:
 *    across an interface the curl-free part is not strictly "the P wave".  In
 *    smooth media and away from interfaces the two coincide.
 *  - The accumulators exist only where lag_gradient.c fills them -- the
 *    model interior for the strain factors (lagBounds), the Vx/Vz loop range
 *    for the density factors.  The split sees each field TRUNCATED there, so
 *    the parts are perturbed within roughly a wavelength of the model edges
 *    (and, for the strain factors, of the top edge near the source).  The
 *    decomposition of what IS accumulated stays exact (PP+PS+SP+SS = total).
 *  - The periodic Poisson solve cannot hold a source with nonzero mean.
 *    The net divergence of the truncated field is put back as a uniform
 *    compensating charge in the zero padding, outside the padded grid, so
 *    tr(e_S) = 0 stays exact; the P part shifts by a smooth harmonic field.
 *  - The vector (density) factors are split over the model interior only;
 *    their P part in the absorbing layer is left zero (all "S"), which
 *    lagStrip discards anyway.
 *  - Valid tests: in a homogeneous background an explosive source makes a
 *    pure P forward field, so the forward S fraction in modesep.txt must be
 *    small and the SP/SS terms negligible; the trace residual of every
 *    strain split must be at round-off.
 *
 * COST: per shot, 4 Poisson solves (8 complex 2D FFTs) per frequency, plus
 * ten more accumulator-sized arrays held while the cubes are written
 * (the P parts; S is formed as total - P on the fly).
 */

#ifndef MAX
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#endif

#define MS_P 0
#define MS_S 1
#define MS_T 2

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

/* ---- module state, valid during one lagModeSepWrite call ---- */
static lagStatePar ms;
static float *P_exx, *P_ezz, *P_shr, *P_pxx, *P_pzz, *P_pxz;
static float *P_dsx, *P_dsz, *P_pvx, *P_pvz;
static int    msNx, msNz;
static float  c1, c2, c3, c4;
/* Projector symbols on the FFT box (index ix*msNz + iz), forward sign -1:
 *   Sxx = D+x D-x / L,  Szz = D+z D-z / L,  Sxz = 2 D-x D-z / L,
 *   Vx  = D-x / L,      Vz  = D-z / L,      L = D+x D-x + D+z D-z.
 * Each 1D symbol is the FFT of the code's own stencil applied to a delta,
 * so the half-cell phases follow whatever sign convention genfft uses (the
 * convolution theorem holds for either).  All five are bounded, so nothing
 * of the 1/k^2 range of the potential ever passes through single precision.
 * The FFT round trip is unscaled; 1/(Nx Nz) is folded into the symbols. */
static complex *msSxx, *msSzz, *msSxz, *msVx, *msVz, *msB1;


static int msPow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

static inline float msGet(const float *T, const float *P, size_t o, int mode)
{
	if (mode == MS_P) return P[o];
	if (mode == MS_S) return T[o] - P[o];
	return T[o];
}

static inline complex msMul(complex a, complex b)
{ complex c; c.r = a.r*b.r - a.i*b.i; c.i = a.r*b.i + a.i*b.r; return c; }

/* Value at offset i of the D+ (plus=1) or D- (plus=0) stencil applied to a
 * unit delta at 0 -- exactly msDp/msDm below. */
static float msStencilResp(int i, int plus)
{
	float c[4]; int m; float v = 0.0f;
	c[0] = c1; c[1] = c2; c[2] = c3; c[3] = c4;
	for (m = 0; m < 4; m++) {
		/* D+ : c_m ( f[i+m+1] - f[i-m] ),  D- : c_m ( f[i+m] - f[i-m-1] ) */
		int a = plus ? i + m + 1 : i + m;
		int b = plus ? i - m     : i - m - 1;
		v += c[m] * ((a == 0 ? 1.0f : 0.0f) - (b == 0 ? 1.0f : 0.0f));
	}
	return v;
}

/* FFT of a 1D stencil laid along x (alongx=1) or z, scaled by 1/d. */
static void msSymbol1D(complex *out, int alongx, int plus, float d)
{
	int i;
	memset(out, 0, (size_t)msNx * msNz * sizeof(complex));
	for (i = -5; i <= 5; i++) {
		float v = msStencilResp(i, plus) / d;
		if (v == 0.0f) continue;
		if (alongx) out[(size_t)((i + msNx) % msNx) * msNz].r = v;
		else        out[(size_t)((i + msNz) % msNz)].r = v;
	}
	cc2dfft(out, msNz, msNx, msNz, -1);
}

static void msBuildSymbols(modPar *mod)
{
	size_t j, nb = (size_t)msNx * msNz;
	float scl = 1.0f / ((float)msNx * (float)msNz);
	complex *Dpx = (complex *)malloc(nb * sizeof(complex));
	complex *Dmx = (complex *)malloc(nb * sizeof(complex));
	complex *Dpz = (complex *)malloc(nb * sizeof(complex));
	complex *Dmz = (complex *)malloc(nb * sizeof(complex));
	msSxx = (complex *)malloc(nb * sizeof(complex));
	msSzz = (complex *)malloc(nb * sizeof(complex));
	msSxz = (complex *)malloc(nb * sizeof(complex));
	msVx  = (complex *)malloc(nb * sizeof(complex));
	msVz  = (complex *)malloc(nb * sizeof(complex));
	msB1  = (complex *)malloc(nb * sizeof(complex));
	if (!Dpx || !Dmx || !Dpz || !Dmz || !msSxx || !msSzz || !msSxz ||
	    !msVx || !msVz || !msB1)
		verr("lagModeSepWrite: cannot allocate the projector symbols.");
	msSymbol1D(Dpx, 1, 1, mod->dx);  msSymbol1D(Dmx, 1, 0, mod->dx);
	msSymbol1D(Dpz, 0, 1, mod->dz);  msSymbol1D(Dmz, 0, 0, mod->dz);

#pragma omp parallel for private(j) schedule(static)
	for (j = 0; j < nb; j++) {
		complex xx = msMul(Dpx[j], Dmx[j]), zz = msMul(Dpz[j], Dmz[j]);
		complex xz = msMul(Dmx[j], Dmz[j]);
		float L = xx.r + zz.r;    /* real: D+ D- is symmetric negative semi-definite */
		float f = (fabsf(L) > 0.0f && j != 0) ? scl / L : 0.0f;
		msSxx[j].r = xx.r * f;        msSxx[j].i = xx.i * f;
		msSzz[j].r = zz.r * f;        msSzz[j].i = zz.i * f;
		msSxz[j].r = 2.0f * xz.r * f; msSxz[j].i = 2.0f * xz.i * f;
		msVx[j].r  = Dmx[j].r * f;    msVx[j].i  = Dmx[j].i * f;
		msVz[j].r  = Dmz[j].r * f;    msVz[j].i  = Dmz[j].i * f;
	}
	free(Dpx); free(Dmx); free(Dpz); free(Dmz);
}

static void msFreeSymbols(void)
{
	free(msSxx); free(msSzz); free(msSxz); free(msVx); free(msVz); free(msB1);
	msSxx = msSzz = msSxz = msVx = msVz = msB1 = NULL;
}

/* Staggered first differences on a COMPLEX padded-grid array a (ld n1).
 * Dp: c1(a[i+1]-a[i]) + c2(a[i+2]-a[i-1]) + ...   (D+, e.g. Vx -> P)
 * s is the stride along the differentiated axis (n1 for x, 1 for z).    */
static inline void msDp(const complex *a, size_t ig, size_t s, float sd,
                        float *re, float *im)
{
	float r = c1*(a[ig+s].r - a[ig].r) + c2*(a[ig+2*s].r - a[ig-s].r)
	        + c3*(a[ig+3*s].r - a[ig-2*s].r) + c4*(a[ig+4*s].r - a[ig-3*s].r);
	float i = c1*(a[ig+s].i - a[ig].i) + c2*(a[ig+2*s].i - a[ig-s].i)
	        + c3*(a[ig+3*s].i - a[ig-2*s].i) + c4*(a[ig+4*s].i - a[ig-3*s].i);
	*re = sd * r;  *im = sd * i;
}

/***********************************************************************
 * msProject -- theta (padded-grid complex, P grid) -> up to three outputs
 *              out_k = IFFT( sym_k * FFT(theta) ) on the padded grid.
 *
 * A field truncated at the edge of the accumulation region has a nonzero
 * net divergence Q, and no periodic Helmholtz projection can carry a source
 * with nonzero mean (k = 0).  Dropping k = 0 would leave tr(e_S) = -Q/Nbox
 * everywhere -- measured 6e-4 / 3e-3 relative on the test, shrinking in
 * proportion to the box.  Instead -Q is put back as a uniform charge in the
 * zero padding, OUTSIDE the padded grid: the source is then exactly zero
 * mean, the projection is exact on the grid, and the P part changes only by
 * a smooth harmonic field (the gauge freedom of a Helmholtz split on a
 * bounded domain).
 ***********************************************************************/
static void msProject(modPar *mod, const complex *theta, complex *box,
                      int nout, complex **sym, complex **out)
{
	int ix, iz, n1 = mod->naz, nax = mod->nax, m;
	size_t j, nb = (size_t)msNx * msNz;
	double qr = 0.0, qi = 0.0;
	size_t npad = nb - (size_t)nax * n1;

	memset(box, 0, nb * sizeof(complex));
#pragma omp parallel for private(ix, iz) reduction(+:qr,qi) schedule(static)
	for (ix = 0; ix < nax; ix++)
		for (iz = 0; iz < n1; iz++) {
			complex t = theta[(size_t)ix * n1 + iz];
			box[(size_t)ix * msNz + iz] = t;
			qr += t.r;  qi += t.i;
		}
	if (npad > 0) {
		float cr = (float)(-qr / (double)npad), ci = (float)(-qi / (double)npad);
#pragma omp parallel for private(ix, iz) schedule(static)
		for (ix = 0; ix < msNx; ix++)
			for (iz = 0; iz < msNz; iz++)
				if (ix >= nax || iz >= n1) {
					box[(size_t)ix * msNz + iz].r = cr;
					box[(size_t)ix * msNz + iz].i = ci;
				}
	}
	cc2dfft(box, msNz, msNx, msNz, -1);

	for (m = 0; m < nout; m++) {
		const complex *S = sym[m];
#pragma omp parallel for private(j) schedule(static)
		for (j = 0; j < nb; j++) msB1[j] = msMul(box[j], S[j]);
		cc2dfft(msB1, msNz, msNx, msNz, 1);
#pragma omp parallel for private(ix, iz) schedule(static)
		for (ix = 0; ix < nax; ix++)
			for (iz = 0; iz < n1; iz++)
				out[m][(size_t)ix * n1 + iz] = msB1[(size_t)ix * msNz + iz];
	}
}

/* Energy bookkeeping for the report: |T|^2, |P|^2, |S|^2 over one field. */
typedef struct { double t, p, s; } msEn;

/***********************************************************************
 * msSplitStrain -- P part of a strain factor (e_xx, e_zz, shear), freq k.
 *
 * Reads the totals Txx, Tzz, Tsh from the lag_gradient accumulators and
 * writes the P parts into Pxx, Pzz, Psh, on exactly the points the totals
 * were accumulated on (lagBounds P and Txz regions).  Also returns the
 * trace residual |tr(e_S)|^2 / |tr(e)|^2 contributions for the self-test.
 ***********************************************************************/
static void msSplitStrain(modPar *mod, bndPar *bnd, int k,
                          const float *Txx, const float *Tzz, const float *Tsh,
                          float *Pxx, float *Pzz, float *Psh,
                          complex *theta, complex *box, complex *phi,
                          complex *wx, complex *wz,
                          double *tr_res, double *tr_tot)
{
	int ix, iz, n1 = mod->naz, nf = ms.nfreq;
	int ibPx, iePx, ibPz, iePz, ibTx, ieTx, ibTz, ieTz;
	double rr = 0.0, tt = 0.0;

	lagBounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz,
	                    &ibTx, &ieTx, &ibTz, &ieTz);
	/* The 8th-order-wide stencils below read 4 points either side. */
	if (ibPx < 4) ibPx = 4;
	if (ibPz < 4) ibPz = 4;
	if (ibTx < 4) ibTx = 4;
	if (ibTz < 4) ibTz = 4;
	if (iePx > mod->nax - 4) iePx = mod->nax - 4;
	if (ieTx > mod->nax - 4) ieTx = mod->nax - 4;
	if (iePz > n1 - 4) iePz = n1 - 4;
	if (ieTz > n1 - 4) ieTz = n1 - 4;

	memset(theta, 0, ms.sizem * sizeof(complex));
#pragma omp parallel for private(ix, iz) schedule(static)
	for (ix = ibPx; ix < iePx; ix++)
		for (iz = ibPz; iz < iePz; iz++) {
			size_t ig = (size_t)ix * n1 + iz, o = ((size_t)ig * nf + k) * 2;
			theta[ig].r = Txx[o]   + Tzz[o];
			theta[ig].i = Txx[o+1] + Tzz[o+1];
		}

	{
		complex *sym[3], *out[3];
		sym[0] = msSxx; sym[1] = msSzz; sym[2] = msSxz;
		out[0] = wx;    out[1] = wz;    out[2] = phi;
		msProject(mod, theta, box, 3, sym, out);
	}

#pragma omp parallel for private(ix, iz) reduction(+:rr,tt) schedule(static)
	for (ix = ibPx; ix < iePx; ix++)
		for (iz = ibPz; iz < iePz; iz++) {
			size_t ig = (size_t)ix * n1 + iz, o = ((size_t)ig * nf + k) * 2;
			float xr = wx[ig].r, xi = wx[ig].i, zr = wz[ig].r, zi = wz[ig].i, sr, si;
			Pxx[o] = xr;  Pxx[o+1] = xi;
			Pzz[o] = zr;  Pzz[o+1] = zi;
			sr = theta[ig].r - (xr + zr);      /* tr(e_S) */
			si = theta[ig].i - (xi + zi);
			rr += (double)sr*sr + (double)si*si;
			tt += (double)theta[ig].r*theta[ig].r + (double)theta[ig].i*theta[ig].i;
		}

#pragma omp parallel for private(ix, iz) schedule(static)
	for (ix = ibTx; ix < ieTx; ix++)
		for (iz = ibTz; iz < ieTz; iz++) {
			size_t ig = (size_t)ix * n1 + iz, o = ((size_t)ig * nf + k) * 2;
			Psh[o] = phi[ig].r;  Psh[o+1] = phi[ig].i;
		}
	(void)Tsh;
	*tr_res += rr;  *tr_tot += tt;
}

/***********************************************************************
 * msSplitVector -- P part of a vector factor (fx on Vx, fz on Vz), freq k.
 *
 * bx, bz: optional pointwise weights mapping the stored field to the
 * physical one before the split (and back after it): the buoyancies for
 * the adjoint psi_v, NULL for the forward velocity increment.
 * Regions are the density accumulators' (ioXx..ieXx etc.).
 ***********************************************************************/
static void msSplitVector(modPar *mod, bndPar *bnd, int k,
                          const float *Tx, const float *Tz,
                          float *Px, float *Pz,
                          const float *bx, const float *bz,
                          complex *fx, complex *fz,
                          complex *theta, complex *box, complex *phi,
                          complex *wx, complex *wz)
{
	int ix, iz, n1 = mod->naz, nax = mod->nax, nf = ms.nfreq;
	int ibPx, iePx, ibPz, iePz, ibTx, ieTx, ibTz, ieTz;
	float sdx = 1.0f / mod->dx, sdz = 1.0f / mod->dz;

	/* Only the model interior goes into the split: the density loops also
	 * run through the absorbing layer, where the field is not a physical
	 * wave and only pollutes the projection.  That part of the gradient
	 * is discarded by lagStrip anyway; there the P part is left zero. */
	lagBounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz,
	                    &ibTx, &ieTx, &ibTz, &ieTz);

	memset(fx, 0, ms.sizem * sizeof(complex));
	memset(fz, 0, ms.sizem * sizeof(complex));
#pragma omp parallel for private(ix, iz) schedule(static)
	for (ix = MAX(mod->ioXx, ibPx); ix < MIN(mod->ieXx, iePx); ix++)
		for (iz = MAX(mod->ioXz, ibPz); iz < MIN(mod->ieXz, iePz); iz++) {
			size_t ig = (size_t)ix * n1 + iz, o = ((size_t)ig * nf + k) * 2;
			float b = bx ? bx[ig] : 1.0f;
			fx[ig].r = b * Tx[o];  fx[ig].i = b * Tx[o+1];
		}
#pragma omp parallel for private(ix, iz) schedule(static)
	for (ix = MAX(mod->ioZx, ibPx); ix < MIN(mod->ieZx, iePx); ix++)
		for (iz = MAX(mod->ioZz, ibPz); iz < MIN(mod->ieZz, iePz); iz++) {
			size_t ig = (size_t)ix * n1 + iz, o = ((size_t)ig * nf + k) * 2;
			float b = bz ? bz[ig] : 1.0f;
			fz[ig].r = b * Tz[o];  fz[ig].i = b * Tz[o+1];
		}

	memset(theta, 0, ms.sizem * sizeof(complex));
#pragma omp parallel for private(ix, iz) schedule(static)
	for (ix = 4; ix < nax - 4; ix++)
		for (iz = 4; iz < n1 - 4; iz++) {
			size_t ig = (size_t)ix * n1 + iz;
			float xr, xi, zr, zi;
			msDp(fx, ig, (size_t)n1, sdx, &xr, &xi);
			msDp(fz, ig, 1,          sdz, &zr, &zi);
			theta[ig].r = xr + zr;  theta[ig].i = xi + zi;
		}

	{
		complex *sym[2], *out[2];
		sym[0] = msVx; sym[1] = msVz;
		out[0] = wx;   out[1] = wz;
		msProject(mod, theta, box, 2, sym, out);
	}
	(void)phi;

#pragma omp parallel for private(ix, iz) schedule(static)
	for (ix = MAX(mod->ioXx, ibPx); ix < MIN(mod->ieXx, iePx); ix++)
		for (iz = MAX(mod->ioXz, ibPz); iz < MIN(mod->ieXz, iePz); iz++) {
			size_t ig = (size_t)ix * n1 + iz, o = ((size_t)ig * nf + k) * 2;
			float b = bx ? bx[ig] : 1.0f;
			float ib = (b != 0.0f) ? 1.0f / b : 0.0f;
			Px[o] = ib * wx[ig].r;  Px[o+1] = ib * wx[ig].i;
		}
#pragma omp parallel for private(ix, iz) schedule(static)
	for (ix = MAX(mod->ioZx, ibPx); ix < MIN(mod->ieZx, iePx); ix++)
		for (iz = MAX(mod->ioZz, ibPz); iz < MIN(mod->ieZz, iePz); iz++) {
			size_t ig = (size_t)ix * n1 + iz, o = ((size_t)ig * nf + k) * 2;
			float b = bz ? bz[ig] : 1.0f;
			float ib = (b != 0.0f) ? 1.0f / b : 0.0f;
			Pz[o] = ib * wz[ig].r;  Pz[o+1] = ib * wz[ig].i;
		}
}

/* |T|^2, |P|^2, |S|^2 of one accumulated field over the points it lives on. */
static void msEnergy(modPar *mod, int ib_x, int ie_x, int ib_z, int ie_z,
                     const float *T, const float *P, msEn *e)
{
	int ix, iz, k, n1 = mod->naz, nf = ms.nfreq;
	double t = 0.0, p = 0.0, s = 0.0;
#pragma omp parallel for private(ix, iz, k) reduction(+:t,p,s) schedule(static)
	for (ix = ib_x; ix < ie_x; ix++)
		for (iz = ib_z; iz < ie_z; iz++) {
			size_t o0 = ((size_t)ix * n1 + iz) * nf * 2;
			for (k = 0; k < 2 * nf; k++) {
				double a = T[o0+k], b = P[o0+k];
				t += a*a;  p += b*b;  s += (a-b)*(a-b);
			}
		}
	e->t += t;  e->p += p;  e->s += s;
}

/***********************************************************************
 * msSlice -- g_lam, g_mu, g_rho_direct at one lag for one mode pair.
 *
 * lagSlice + lagSliceRho with every factor fetched as its P part, S part
 * or total (mf for the forward factor, ma for the adjoint one).  The
 * frequency sums run in parallel into scratch, and only the scatter of the
 * Txz and Vx/Vz contributions onto P points is serial.
 ***********************************************************************/
static void msSlice(modPar *mod, bndPar *bnd, float tau, int mf, int ma,
                    float *glam, float *gmuu, float *grho, float *scr,
                    const float *ct, const float *st)
{
	int ix, iz, k, n1 = mod->naz, nf = ms.nfreq;
	int ibPx, iePx, ibPz, iePz, ibTx, ieTx, ibTz, ieTz;
	float w  = (ms.df > 0.0f) ? 2.0f * ms.df : 1.0f;
	float wr = (ms.dt > 0.0f) ? w / ms.dt : w;
	(void)tau;

	lagBounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz,
	                    &ibTx, &ieTx, &ibTz, &ieTz);
	memset(glam, 0, ms.sizem * sizeof(float));
	memset(gmuu, 0, ms.sizem * sizeof(float));
	memset(grho, 0, ms.sizem * sizeof(float));

	/* P grid: lambda and the normal half of mu */
#pragma omp parallel for private(ix, iz, k) schedule(static)
	for (ix = ibPx; ix < iePx; ix++)
		for (iz = ibPz; iz < iePz; iz++) {
			size_t ig = (size_t)ix * n1 + iz, o0 = ig * nf * 2;
			float al = 0.0f, am = 0.0f;
			for (k = 0; k < nf; k++) {
				size_t o = o0 + (size_t)k * 2;
				float exr = msGet(ms.exx, P_exx, o, mf), exi = msGet(ms.exx, P_exx, o+1, mf);
				float ezr = msGet(ms.ezz, P_ezz, o, mf), ezi = msGet(ms.ezz, P_ezz, o+1, mf);
				float pxr = msGet(ms.pxx, P_pxx, o, ma), pxi = msGet(ms.pxx, P_pxx, o+1, ma);
				float pzr = msGet(ms.pzz, P_pzz, o, ma), pzi = msGet(ms.pzz, P_pzz, o+1, ma);
				float dr = exr + ezr, di = exi + ezi, sr = pxr + pzr, si = pxi + pzi;
				al += (dr*sr + di*si)*ct[k] + (dr*si - di*sr)*st[k];
				am += 2.0f*(((exr*pxr + exi*pxi) + (ezr*pzr + ezi*pzi))*ct[k]
				          + ((exr*pxi - exi*pxr) + (ezr*pzi - ezi*pzr))*st[k]);
			}
			glam[ig] = w * al;
			gmuu[ig] = w * am;
		}

	/* Txz grid: shear half of mu, 0.25 to the four surrounding P points */
	memset(scr, 0, ms.sizem * sizeof(float));
#pragma omp parallel for private(ix, iz, k) schedule(static)
	for (ix = ibTx; ix < ieTx; ix++)
		for (iz = ibTz; iz < ieTz; iz++) {
			size_t ig = (size_t)ix * n1 + iz, o0 = ig * nf * 2;
			float a = 0.0f;
			for (k = 0; k < nf; k++) {
				size_t o = o0 + (size_t)k * 2;
				float sr = msGet(ms.shr, P_shr, o, mf), si = msGet(ms.shr, P_shr, o+1, mf);
				float pr = msGet(ms.pxz, P_pxz, o, ma), pi = msGet(ms.pxz, P_pxz, o+1, ma);
				a += (sr*pr + si*pi)*ct[k] + (sr*pi - si*pr)*st[k];
			}
			scr[ig] = w * a;
		}
	for (ix = ibTx; ix < ieTx; ix++)
		for (iz = ibTz; iz < ieTz; iz++) {
			size_t ig = (size_t)ix * n1 + iz;
			float s = 0.25f * scr[ig];
			gmuu[(ix-1)*n1+iz-1] += s;  gmuu[ix*n1+iz-1] += s;
			gmuu[(ix-1)*n1+iz]   += s;  gmuu[ig]         += s;
		}

	/* density, Vx grid: 0.5/rho to the two P points vx straddles */
	memset(scr, 0, ms.sizem * sizeof(float));
#pragma omp parallel for private(ix, iz, k) schedule(static)
	for (ix = mod->ioXx; ix < mod->ieXx; ix++)
		for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
			size_t ig = (size_t)ix * n1 + iz, o0 = ig * nf * 2;
			float a = 0.0f;
			for (k = 0; k < nf; k++) {
				size_t o = o0 + (size_t)k * 2;
				float ar = msGet(ms.dsx, P_dsx, o, mf), ai = msGet(ms.dsx, P_dsx, o+1, mf);
				float br = msGet(ms.pvx, P_pvx, o, ma), bi = msGet(ms.pvx, P_pvx, o+1, ma);
				a += (ar*br + ai*bi)*ct[k] + (ar*bi - ai*br)*st[k];
			}
			scr[ig] = wr * a;
		}
	for (ix = mod->ioXx; ix < mod->ieXx; ix++)
		for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
			size_t ig = (size_t)ix * n1 + iz;
			grho[(ix-1)*n1+iz] += 0.5f*scr[ig] / mod->rho[(ix-1)*n1+iz];
			grho[ig]           += 0.5f*scr[ig] / mod->rho[ig];
		}

	/* density, Vz grid */
	memset(scr, 0, ms.sizem * sizeof(float));
#pragma omp parallel for private(ix, iz, k) schedule(static)
	for (ix = mod->ioZx; ix < mod->ieZx; ix++)
		for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
			size_t ig = (size_t)ix * n1 + iz, o0 = ig * nf * 2;
			float a = 0.0f;
			for (k = 0; k < nf; k++) {
				size_t o = o0 + (size_t)k * 2;
				float ar = msGet(ms.dsz, P_dsz, o, mf), ai = msGet(ms.dsz, P_dsz, o+1, mf);
				float br = msGet(ms.pvz, P_pvz, o, ma), bi = msGet(ms.pvz, P_pvz, o+1, ma);
				a += (ar*br + ai*bi)*ct[k] + (ar*bi - ai*br)*st[k];
			}
			scr[ig] = wr * a;
		}
	for (ix = mod->ioZx; ix < mod->ieZx; ix++)
		for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
			size_t ig = (size_t)ix * n1 + iz;
			grho[ig-1] += 0.5f*scr[ig] / mod->rho[ig-1];
			grho[ig]   += 0.5f*scr[ig] / mod->rho[ig];
		}
}

/* Chain rule to (Vp, Vs, rho), interior strip, taper -- as lagBuildCubes. */
static void msConvert(modPar *mod, bndPar *bnd, const float *glam,
                      const float *gmuu, const float *grho, float *slice,
                      float *mask, float *ovp, float *ovs, float *oro)
{
	size_t i, nmodel = (size_t)mod->nx * mod->nz;
	if (ovp) {
		for (i = 0; i < ms.sizem; i++)
			slice[i] = 2.0f * mod->rho[i] * mod->cp[i] * glam[i];
		lagStrip(ovp, slice, mod, bnd);
	}
	for (i = 0; i < ms.sizem; i++)
		slice[i] = 2.0f * mod->rho[i] * mod->cs[i] * (gmuu[i] - 2.0f * glam[i]);
	lagStrip(ovs, slice, mod, bnd);
	for (i = 0; i < ms.sizem; i++) {
		float vp2 = mod->cp[i]*mod->cp[i], vs2 = mod->cs[i]*mod->cs[i];
		slice[i] = grho[i] + glam[i]*(vp2 - 2.0f*vs2) + gmuu[i]*vs2;
	}
	lagStrip(oro, slice, mod, bnd);
	if (mask) {
		for (i = 0; i < nmodel; i++) {
			if (ovp) ovp[i] *= mask[i];
			ovs[i] *= mask[i];  oro[i] *= mask[i];
		}
	}
}

static double msNorm(const float *a, size_t n)
{ size_t i; double s = 0.0; for (i = 0; i < n; i++) s += (double)a[i]*a[i]; return sqrt(s); }
static double msDiffNorm(const float *a, const float *b, size_t n)
{ size_t i; double s = 0.0; for (i = 0; i < n; i++) { double d = (double)a[i]-b[i]; s += d*d; } return sqrt(s); }


/***********************************************************************
 * lagModeSepWrite -- see fdelfwi.h and the file header.
 ***********************************************************************/
void lagModeSepWrite(modPar *mod, bndPar *bnd, const char *base, int ishot,
                     int nlag, float dlag, float *mask, int verbose)
{
	int k, ilag, mid, nf, c;
	size_t nbytes, nmodel, i;
	complex *theta, *box, *phi, *wx, *wz, *fx, *fz;
	float *glam, *gmuu, *grho, *scr, *slice, *ct, *st;
	float *cPPvs, *cPPro, *cSSvs, *cSSro;
	double tr_res_f = 0.0, tr_tot_f = 0.0, tr_res_a = 0.0, tr_tot_a = 0.0;
	msEn e_fs = {0,0,0}, e_as = {0,0,0}, e_fv = {0,0,0}, e_av = {0,0,0};
	int ibPx, iePx, ibPz, iePz, ibTx, ieTx, ibTz, ieTz;
	char fname[1024];
	FILE *fp;
	double t0 = wallclock_time();

	lagGradientState(&ms);
	if (!ms.on) return;
	if (!ms.rho_on || ms.rho_nacc == 0)
		verr("lagModeSepWrite: needs the density accumulators (lag_rho=1) and "
		     "a run that fed them (rho_nacc=%d).", ms.rho_nacc);
	nlag = lagAxisCheck("lagModeSepWrite", nlag, dlag);
	if (nlag == 0) return;

	nf     = ms.nfreq;
	nmodel = (size_t)mod->nx * mod->nz;
	nbytes = ms.sizem * (size_t)nf * 2 * sizeof(float);
	lagCoef(mod, &c1, &c2, &c3, &c4);

	/* ---- P-part accumulators ---- */
	P_exx = (float *)calloc(1, nbytes);  P_ezz = (float *)calloc(1, nbytes);
	P_shr = (float *)calloc(1, nbytes);  P_pxx = (float *)calloc(1, nbytes);
	P_pzz = (float *)calloc(1, nbytes);  P_pxz = (float *)calloc(1, nbytes);
	P_dsx = (float *)calloc(1, nbytes);  P_dsz = (float *)calloc(1, nbytes);
	P_pvx = (float *)calloc(1, nbytes);  P_pvz = (float *)calloc(1, nbytes);
	if (!P_exx || !P_ezz || !P_shr || !P_pxx || !P_pzz || !P_pxz ||
	    !P_dsx || !P_dsz || !P_pvx || !P_pvz)
		verr("lagModeSepWrite: cannot allocate 10 x %.1f MB for the P parts.",
		     nbytes / 1048576.0);

	/* ---- FFT box and symbols ---- */
	{
		/* Zero padding of the periodic Poisson box beyond the padded grid
		 * (diagnostic knob; see "WHAT IS NOT EXACT"). */
		int pad;
		if (!getparint("lag_modesep_pad", &pad)) pad = 8;
		msNx = msPow2(mod->nax + pad);
		msNz = msPow2(mod->naz + pad);
	}
	msBuildSymbols(mod);

	box   = (complex *)malloc((size_t)msNx * msNz * sizeof(complex));
	theta = (complex *)malloc(ms.sizem * sizeof(complex));
	phi   = (complex *)malloc(ms.sizem * sizeof(complex));
	wx    = (complex *)malloc(ms.sizem * sizeof(complex));
	wz    = (complex *)malloc(ms.sizem * sizeof(complex));
	fx    = (complex *)malloc(ms.sizem * sizeof(complex));
	fz    = (complex *)malloc(ms.sizem * sizeof(complex));
	if (!box || !theta || !phi || !wx || !wz || !fx || !fz)
		verr("lagModeSepWrite: cannot allocate the projection scratch.");

	if (verbose)
		vmess("lagModeSepWrite: shot %d, %d frequencies, FFT box %d x %d, "
		      "+%.1f MB for the P parts", ishot, nf, msNx, msNz,
		      10.0 * nbytes / 1048576.0);

	/* ---- split every factor, one frequency at a time ---- */
	for (k = 0; k < nf; k++) {
		msSplitStrain(mod, bnd, k, ms.exx, ms.ezz, ms.shr, P_exx, P_ezz, P_shr,
		              theta, box, phi, wx, wz, &tr_res_f, &tr_tot_f);
		msSplitStrain(mod, bnd, k, ms.pxx, ms.pzz, ms.pxz, P_pxx, P_pzz, P_pxz,
		              theta, box, phi, wx, wz, &tr_res_a, &tr_tot_a);
		msSplitVector(mod, bnd, k, ms.dsx, ms.dsz, P_dsx, P_dsz, NULL, NULL,
		              fx, fz, theta, box, phi, wx, wz);
		msSplitVector(mod, bnd, k, ms.pvx, ms.pvz, P_pvx, P_pvz, mod->rox, mod->roz,
		              fx, fz, theta, box, phi, wx, wz);
	}
	free(box); free(theta); free(phi); free(wx); free(wz); free(fx); free(fz);
	msFreeSymbols();
	if (verbose)
		vmess("lagModeSepWrite: split done in %.1f s", wallclock_time() - t0);

	/* ---- energy report: how much of each factor is P, how much S ---- */
	lagBounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz, &ibTx, &ieTx, &ibTz, &ieTz);
	msEnergy(mod, ibPx, iePx, ibPz, iePz, ms.exx, P_exx, &e_fs);
	msEnergy(mod, ibPx, iePx, ibPz, iePz, ms.ezz, P_ezz, &e_fs);
	msEnergy(mod, ibTx, ieTx, ibTz, ieTz, ms.shr, P_shr, &e_fs);
	msEnergy(mod, ibPx, iePx, ibPz, iePz, ms.pxx, P_pxx, &e_as);
	msEnergy(mod, ibPx, iePx, ibPz, iePz, ms.pzz, P_pzz, &e_as);
	msEnergy(mod, ibTx, ieTx, ibTz, ieTz, ms.pxz, P_pxz, &e_as);
	/* vector factors: interior only, where they were split */
	msEnergy(mod, MAX(mod->ioXx,ibPx), MIN(mod->ieXx,iePx), MAX(mod->ioXz,ibPz), MIN(mod->ieXz,iePz), ms.dsx, P_dsx, &e_fv);
	msEnergy(mod, MAX(mod->ioZx,ibPx), MIN(mod->ieZx,iePx), MAX(mod->ioZz,ibPz), MIN(mod->ieZz,iePz), ms.dsz, P_dsz, &e_fv);
	msEnergy(mod, MAX(mod->ioXx,ibPx), MIN(mod->ieXx,iePx), MAX(mod->ioXz,ibPz), MIN(mod->ieXz,iePz), ms.pvx, P_pvx, &e_av);
	msEnergy(mod, MAX(mod->ioZx,ibPx), MIN(mod->ieZx,iePx), MAX(mod->ioZz,ibPz), MIN(mod->ieZz,iePz), ms.pvz, P_pvz, &e_av);

	/* ---- scratch for the reconstruction ---- */
	glam  = (float *)malloc(ms.sizem * sizeof(float));
	gmuu  = (float *)malloc(ms.sizem * sizeof(float));
	grho  = (float *)malloc(ms.sizem * sizeof(float));
	scr   = (float *)malloc(ms.sizem * sizeof(float));
	slice = (float *)malloc(ms.sizem * sizeof(float));
	ct    = (float *)malloc((size_t)nf * sizeof(float));
	st    = (float *)malloc((size_t)nf * sizeof(float));
	if (!glam || !gmuu || !grho || !scr || !slice || !ct || !st)
		verr("lagModeSepWrite: cannot allocate slice scratch.");

	/* ---- tau = 0 report: closure, PP/SS/cross shares, Vp identity ---- */
	{
		float *vp[5], *vs[5], *ro[5];   /* PP, PS, SP, SS, total */
		int pair[5][2] = {{MS_P,MS_P},{MS_P,MS_S},{MS_S,MS_P},{MS_S,MS_S},{MS_T,MS_T}};
		const char *lab[5] = {"PP", "PS", "SP", "SS", "total"};
		double nvp, nvs, nro;
		float *svp, *svs, *sro;

		for (k = 0; k < nf; k++) { ct[k] = 1.0f; st[k] = 0.0f; }
		for (c = 0; c < 5; c++) {
			vp[c] = (float *)malloc(nmodel * sizeof(float));
			vs[c] = (float *)malloc(nmodel * sizeof(float));
			ro[c] = (float *)malloc(nmodel * sizeof(float));
			if (!vp[c] || !vs[c] || !ro[c]) verr("lagModeSepWrite: tau=0 scratch.");
			msSlice(mod, bnd, 0.0f, pair[c][0], pair[c][1], glam, gmuu, grho,
			        scr, ct, st);
			msConvert(mod, bnd, glam, gmuu, grho, slice, mask, vp[c], vs[c], ro[c]);
		}
		svp = (float *)calloc(nmodel, sizeof(float));
		svs = (float *)calloc(nmodel, sizeof(float));
		sro = (float *)calloc(nmodel, sizeof(float));
		for (c = 0; c < 4; c++)
			for (i = 0; i < nmodel; i++) {
				svp[i] += vp[c][i]; svs[i] += vs[c][i]; sro[i] += ro[c][i];
			}
		nvp = msNorm(vp[4], nmodel); nvs = msNorm(vs[4], nmodel); nro = msNorm(ro[4], nmodel);

		snprintf(fname, sizeof(fname), "%s_shot%03d_modesep.txt", base, ishot);
		fp = fopen(fname, "w");
		if (!fp) verr("lagModeSepWrite: cannot write %s", fname);
		fprintf(fp, "# P/S mode separation, shot %d (lag_modesep.c)\n", ishot);
		fprintf(fp, "# FFT box %d x %d, %d frequencies\n", msNx, msNz, nf);
		fprintf(fp, "#\n# factor energy (all frequencies), fraction of |total|^2\n");
		fprintf(fp, "#   field                          P        S        (P+S-1: cross)\n");
#define MSROW(name, e) fprintf(fp, "    %-30s %8.5f %8.5f %+9.2e\n", name, \
		(e).p/((e).t>0?(e).t:1), (e).s/((e).t>0?(e).t:1), ((e).p+(e).s-(e).t)/((e).t>0?(e).t:1))
		MSROW("forward strain rate", e_fs);
		MSROW("adjoint psi_sigma (strain)", e_as);
		MSROW("forward velocity increment", e_fv);
		MSROW("adjoint psi_v", e_av);
		fprintf(fp, "#\n# self-test: S part of a strain is traceless\n");
		fprintf(fp, "    trace residual forward  %.3e\n", tr_tot_f > 0 ? sqrt(tr_res_f/tr_tot_f) : 0.0);
		fprintf(fp, "    trace residual adjoint  %.3e\n", tr_tot_a > 0 ? sqrt(tr_res_a/tr_tot_a) : 0.0);
		fprintf(fp, "#\n# tau = 0 gradient (same taper as the cubes), ||g_mode|| / ||g_total||\n");
		fprintf(fp, "#   mode        Vp        Vs       rho\n");
		for (c = 0; c < 4; c++)
			fprintf(fp, "    %-6s %9.5f %9.5f %9.5f\n", lab[c],
			        nvp > 0 ? msNorm(vp[c], nmodel)/nvp : 0.0,
			        nvs > 0 ? msNorm(vs[c], nmodel)/nvs : 0.0,
			        nro > 0 ? msNorm(ro[c], nmodel)/nro : 0.0);
		fprintf(fp, "#\n# closure ||PP+PS+SP+SS - total|| / ||total||  (round-off expected)\n");
		fprintf(fp, "    closure  Vp %.3e  Vs %.3e  rho %.3e\n",
		        nvp > 0 ? msDiffNorm(svp, vp[4], nmodel)/nvp : 0.0,
		        nvs > 0 ? msDiffNorm(svs, vs[4], nmodel)/nvs : 0.0,
		        nro > 0 ? msDiffNorm(sro, ro[4], nmodel)/nro : 0.0);
		fprintf(fp, "# Vp identity: ||PP - total||/||total|| (Vp is pure PP)\n");
		fprintf(fp, "    vp_pp_vs_total %.3e\n", nvp > 0 ? msDiffNorm(vp[0], vp[4], nmodel)/nvp : 0.0);
		fclose(fp);
		if (verbose) {
			vmess("lagModeSepWrite: tau=0 ||g_Vs||: PP %.4f  PS %.4f  SP %.4f  SS %.4f  of total",
			      msNorm(vs[0],nmodel)/nvs, msNorm(vs[1],nmodel)/nvs,
			      msNorm(vs[2],nmodel)/nvs, msNorm(vs[3],nmodel)/nvs);
			vmess("lagModeSepWrite: closure Vs %.2e rho %.2e, Vp PP-total %.2e, "
			      "trace residual fwd %.2e adj %.2e  -> %s",
			      msDiffNorm(svs, vs[4], nmodel)/nvs, msDiffNorm(sro, ro[4], nmodel)/nro,
			      msDiffNorm(vp[0], vp[4], nmodel)/nvp,
			      tr_tot_f > 0 ? sqrt(tr_res_f/tr_tot_f) : 0.0,
			      tr_tot_a > 0 ? sqrt(tr_res_a/tr_tot_a) : 0.0, fname);
		}
		for (c = 0; c < 5; c++) { free(vp[c]); free(vs[c]); free(ro[c]); }
		free(svp); free(svs); free(sro);
	}

	/* ---- the PP and SS lag cubes ---- */
	cPPvs = (float *)malloc(nmodel * nlag * sizeof(float));
	cPPro = (float *)malloc(nmodel * nlag * sizeof(float));
	cSSvs = (float *)malloc(nmodel * nlag * sizeof(float));
	cSSro = (float *)malloc(nmodel * nlag * sizeof(float));
	if (!cPPvs || !cPPro || !cSSvs || !cSSro)
		verr("lagModeSepWrite: cannot allocate 4 cubes of %d lags x %zu points.",
		     nlag, nmodel);
	mid = (nlag - 1) / 2;
	for (ilag = 0; ilag < nlag; ilag++) {
		float tau = (float)(ilag - mid) * dlag;
		size_t off = (size_t)ilag * nmodel;
		for (k = 0; k < nf; k++) {
			ct[k] = cosf(2.0f * ms.omega[k] * tau);
			st[k] = sinf(2.0f * ms.omega[k] * tau);
		}
		msSlice(mod, bnd, tau, MS_P, MS_P, glam, gmuu, grho, scr, ct, st);
		msConvert(mod, bnd, glam, gmuu, grho, slice, mask, NULL, cPPvs + off, cPPro + off);
		msSlice(mod, bnd, tau, MS_S, MS_S, glam, gmuu, grho, scr, ct, st);
		msConvert(mod, bnd, glam, gmuu, grho, slice, mask, NULL, cSSvs + off, cSSro + off);
	}

#define MSWRITE(tag, par, cube) \
	snprintf(fname, sizeof(fname), "%s_%s_shot%03d_lag%s.su", base, tag, ishot, par); \
	writesufile(fname, cube, (size_t)mod->nz, (size_t)mod->nx * nlag, \
	            mod->z0, mod->x0, mod->dz, mod->dx);
	MSWRITE("PP", "vs",  cPPvs);
	MSWRITE("PP", "rho", cPPro);
	MSWRITE("SS", "vs",  cSSvs);
	MSWRITE("SS", "rho", cSSro);
	if (verbose)
		vmess("lagModeSepWrite: wrote %s_{PP,SS}_shot%03d_lag{vs,rho}.su "
		      "(%d x %d x %d, lag slowest) in %.1f s total",
		      base, ishot, nlag, mod->nx, mod->nz, wallclock_time() - t0);

	free(cPPvs); free(cPPro); free(cSSvs); free(cSSro);
	free(glam); free(gmuu); free(grho); free(scr); free(slice); free(ct); free(st);
	free(P_exx); free(P_ezz); free(P_shr); free(P_pxx); free(P_pzz); free(P_pxz);
	free(P_dsx); free(P_dsz); free(P_pvx); free(P_pvz);
	P_exx = P_ezz = P_shr = P_pxx = P_pzz = P_pxz = NULL;
	P_dsx = P_dsz = P_pvx = P_pvz = NULL;
}
