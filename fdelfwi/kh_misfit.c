#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"

void vmess(char *fmt, ...);
void vwarn(char *fmt, ...);
void verr(char *fmt, ...);

/*********************************************************************
 *
 * kh_misfit - Kirchhoff-Helmholtz (energy-flux) objective.
 *
 *   J = sum_r w_r sum_n P(r_r^n),   r = (dsig_ij n_j) dv_i = dt_i dv_i
 *
 * with d = syn - obs, n the recording-surface normal and w_r the
 * receiver quadrature weight.  r is the instantaneous energy-flux
 * density of the RESIDUAL wavefield through the recording surface
 * (traction x velocity, Pa*m/s = W/m^2).
 *
 * Only the linear penalty P(r) = r, P'(r) = 1 is implemented.  r is
 * already bilinear in the residual, so J is O(d^2) -- the same order as
 * L2, giving an O(d) gradient and a non-degenerate Gauss-Newton
 * Hessian.  (P = r^2/2 would make J quartic and the minimum degenerate;
 * P = |r| keeps O(d^2) but P' = sign(r) is discontinuous at every zero
 * crossing, so J is differentiable only almost everywhere.)
 *
 * ------------------------------------------------------------------
 * THE ADJOINT SOURCE
 * ------------------------------------------------------------------
 * For J = sum_n g(u^n), the adjoint-state method injects exactly
 * dJ/du^n into the multiplier at step n.  Nothing else: no material
 * weights, no grid factors.  Three facts make that literal here:
 *
 *   - elastic4_adj.c is the true matrix transpose, so the adjoint array
 *     `txx` is dual to the forward `txx`, one-to-one for all five;
 *   - applyAdjointSource.c adds the trace value RAW (`vx[i] += ampl`),
 *     so R^T of a point sample carries unit weight;
 *   - the house convention (computeResidual.c, L2 branch) is that J is a
 *     plain sum over samples with NO dt, and psi = dJ/dd with no dt.
 *
 * For a flat line, n = (0, n_z):
 *
 *   r = n_z * (dsig_xz * dv_x  +  dsig_zz * dv_z)
 *
 * and differentiating J term by term gives the five traces
 *
 *   a_vx  = w_r * n_z * dsig_xz      (trid 6)   <- traction  -> force
 *   a_vz  = w_r * n_z * dsig_zz      (trid 7)   <- traction  -> force
 *   a_tzz = w_r * n_z * dv_z         (trid 3)   <- velocity  -> defm rate
 *   a_txz = w_r * n_z * dv_x         (trid 2)   <- velocity  -> defm rate
 *   a_txx = 0                        (trid 4)
 *
 * sig_xx drops out because it is not part of the traction on a
 * horizontal plane (t_i = sig_iz n_z) and because the xx strain-rate
 * entry is n_x dv_x = 0.  Physically, the tangential normal stress does
 * no work on that plane.  It returns the moment the normal is tilted,
 * through both t_x = sig_xx n_x + sig_xz n_z and eps_xx = n_x dv_x, so
 * recording _rtxx anyway costs nothing and a dipping-normal test then
 * needs no remodelling.
 *
 * There is no factor of 1/2 on the shear slot: dr/dsig_xz = dv_x
 * directly, so the Voigt engineering-vs-tensor-strain question never
 * arises.
 *
 * RELATION TO RAVASI & CURTIS (2013), Geophysics 78(6), S265-S284
 *   Their eq. 7 injects stress recordings as volume force density
 *   sources (the f backpropagator) and velocity recordings as
 *   deformation rate density sources (the h backpropagator) -- the same
 *   traction->force, velocity->deformation split as the table above.
 *   Their eq. 8, injecting particle velocity alone ("vectorial"
 *   extrapolation), is a heuristic they show contradicts reciprocity,
 *   and it produces two artifacts (S269-S271): each arrival radiates
 *   both up and down instead of unidirectionally, and injected energy
 *   emits both P and S modes.  Independent per-component point
 *   injection -- what the L2 adjoint source does -- is worse still.
 *
 *   With P = linear, P' = 1, so this adjoint source IS their tensorial
 *   injection applied to the residual.  The artifact suppression is
 *   therefore obtained as the exact gradient of a stated objective
 *   rather than as a substitution.
 *
 *   Their eq. 11 carries a MINUS between the traction and velocity terms
 *   where dJ/du gives both +.  That is a difference of variables, not of
 *   physics: see SIGN CONVENTION -- fdelmodc's stress arrays are -sigma,
 *   so writing psi in terms of physical stress reintroduces a minus on
 *   the traction slots alone.  The relative sign is not cosmetic (it is
 *   what makes the two terms interfere on one side of the line and
 *   cancel on the other), so it was confirmed by measurement and not by
 *   this argument alone.
 *
 *   MEASURED (tests/kh_misfit/directionality.py, 2026-10-23).  Elastic
 *   energy in the adjoint snapshots, summed over frames, in MIRROR-IMAGE
 *   windows above and below the recording line:
 *
 *                            E_above/E_below      asymmetry
 *     K-H, this file              0.047-0.053       0.90-0.91
 *     L2 on _rvx,_rvz             0.84 -0.97        0.01-0.09
 *
 *   (recording line z=150 m, pad 5/10/15 cells.)  The K-H source puts
 *   ~20x more energy into one half-space; the L2 source splits almost
 *   evenly.  That is exactly the R&C unidirectionality claim, and it is
 *   the reason to prefer this objective's adjoint over L2's: the
 *   backward-radiating half of an L2 injection correlates with the
 *   forward field on the wrong side of the receivers.
 *
 *   n_z DOES NOT STEER THE RADIATION.  Tempting mistake: n_z multiplies
 *   ALL FOUR slots, so flipping it negates the whole adjoint source.  The
 *   wavefield changes sign, the energy does not move, and J and the
 *   gradient flip sign while the radiation stays on the same side
 *   (verified: kh_normal=+1 and -1 give bit-identical energy maps).  The
 *   side is fixed by the RELATIVE sign of the traction and velocity
 *   slots, which dJ/du determines -- there is no free parameter there,
 *   and nothing to tune if the radiation looks wrong.  It would mean the
 *   derivation is wrong.
 *
 *   The radiating side is also NOT data-adaptive: with the line moved
 *   below the anomaly, so the residual arrives downgoing instead of
 *   upgoing, the K-H source still radiates into the same half-space
 *   (ratio 0.24-0.50, so weaker but the same sign).  Unidirectionality is
 *   a property of how the source is built, not of what it is fed.
 *
 * SIGN CONVENTION  (settled -- derived and measured, 2026-10-23)
 *   fdelmodc does NOT store physical stress.  Both kernel updates carry a
 *   minus (elastic4.c:113 `vz -= roz*[D-z(tzz) + D-x(txz)]` and
 *   elastic4.c:146 `txx -= l2m*dvx + lam*dvz`), which is consistent only
 *   with
 *                         T_ij = -sigma_ij
 *
 *   i.e. the arrays are pressure-positive.  The velocities keep their
 *   physical sign; the momentum equation's minus is what absorbs the
 *   stress flip.
 *
 *   The elastic power-flux (Poynting) vector is S_j = -sigma_ij v_i, so
 *   the flux along +n is -t_i v_i in PHYSICAL variables.  In the code's
 *   variables that second minus cancels the storage minus:
 *
 *       flux along +n  =  -(sigma_ij n_j) v_i  =  +(T_ij n_j) v_i  =  +r
 *
 *   So r as computed below IS the energy flux along +n, with no extra
 *   sign, and J > 0 means residual energy crossing the recording surface
 *   in the +n direction.
 *
 *   MEASURED, not just derived.  Same source (z=50 m) and anomaly
 *   (z=250-350 m), n_z=+1 (pointing DOWN, since z increases downward),
 *   only the recording line moved:
 *
 *     line at z=150 m, ABOVE the anomaly   ->  J = -2.330e-01
 *     line at z=400 m, BELOW the anomaly   ->  J = +9.042e+00
 *
 *   The direct wave cancels in syn-obs, so the residual at a line above
 *   the anomaly is purely upgoing and at a line below it is dominantly
 *   downgoing.  J flips sign with the flux direction and is negative for
 *   upgoing residual energy against a downward normal -- exactly as the
 *   identity above requires.  (The 39x magnitude difference is just the
 *   transmitted perturbation being far stronger than the backscatter off
 *   a 15% contrast.)  n_z stays a parameter so a dipping surface or the
 *   opposite orientation needs no recompile.
 *
 * WHY data_weight MUST BE 0
 *   Both weightings rescale velocity channels relative to stress
 *   channels.  r is a product of one of each, so any such rescaling
 *   destroys the traction:velocity balance that carries the energy-flux
 *   meaning -- the result is still a well-defined objective with an
 *   exact gradient, but it is no longer an energy flux.
 *
 **********************************************************************/


/*---------------------------------------------------------------------
 * khWeightsBuild - receiver quadrature weights w_r.
 *
 * Trapezoid rule on the actual receiver positions, so irregular and
 * gapped spacing are handled correctly:
 *
 *   w_0     = (x_1 - x_0)/2
 *   w_i     = (x_{i+1} - x_{i-1})/2
 *   w_{N-1} = (x_{N-1} - x_{N-2})/2
 *
 * Positions must be monotonic: the trapezoid rule is meaningless
 * otherwise and a silently wrong quadrature is near-impossible to spot
 * downstream, so a non-monotonic array is a hard error.
 *-------------------------------------------------------------------*/
khPar *khWeightsBuild(recPar *rec, float normal_z, int verbose)
{
	khPar *kh;
	int    i, n;
	double wsum = 0.0;

	if (!rec) verr("khWeightsBuild: NULL recPar.");
	n = rec->n;
	if (n < 1) verr("khWeightsBuild: no receivers.");
	if (!rec->xr) verr("khWeightsBuild: receiver x positions are not set.");

	if (normal_z == 0.0f)
		verr("khWeightsBuild: kh_normal=0 -- the recording surface normal "
		     "must have a non-zero z component for a horizontal line.");

	kh = (khPar *)malloc(sizeof(khPar));
	if (!kh) verr("khWeightsBuild: allocation failed.");
	kh->nrec = n;
	kh->nz   = normal_z;
	kh->w    = (float *)malloc((size_t)n * sizeof(float));
	if (!kh->w) verr("khWeightsBuild: allocation failed for %d weights.", n);

	if (n == 1) {
		/* A single receiver has no spacing to integrate over.  Unit
		 * weight keeps single-receiver impulse-response diagnostics
		 * usable; J is then a flux density, not a flux. */
		kh->w[0] = 1.0f;
		if (verbose)
			vmess("khWeightsBuild: 1 receiver, quadrature weight set to 1.0 "
			      "(J is a flux DENSITY in W/m^2, not a flux).");
		return kh;
	}

	for (i = 1; i < n; i++) {
		if (rec->xr[i] <= rec->xr[i-1])
			verr("khWeightsBuild: receiver x positions must be strictly "
			     "increasing for the trapezoid quadrature, but xr[%d]=%.3f "
			     "is not greater than xr[%d]=%.3f.",
			     i, rec->xr[i], i-1, rec->xr[i-1]);
	}

	kh->w[0]   = 0.5f * (rec->xr[1] - rec->xr[0]);
	kh->w[n-1] = 0.5f * (rec->xr[n-1] - rec->xr[n-2]);
	for (i = 1; i < n-1; i++)
		kh->w[i] = 0.5f * (rec->xr[i+1] - rec->xr[i-1]);

	for (i = 0; i < n; i++) wsum += kh->w[i];

	if (verbose)
		vmess("khWeightsBuild: %d receivers, n_z=%+.1f, aperture %.1f m, "
		      "sum(w_r)=%.3f m.", n, normal_z,
		      rec->xr[n-1] - rec->xr[0], wsum);

	return kh;
}


void khWeightsFree(khPar *kh)
{
	if (!kh) return;
	free(kh->w);
	free(kh);
}


/*---------------------------------------------------------------------
 * khResolveChannels - locate the required components in the trid array.
 *
 * Fills idx[] with the position of each needed channel, or -1.  Called
 * once per gather so the per-sample loop stays branch-free.
 *-------------------------------------------------------------------*/
void khResolveChannels(int ncomp, const int *chan, int *idx_vx, int *idx_vz,
                       int *idx_txz, int *idx_tzz, int *idx_txx)
{
	int k;

	*idx_vx = *idx_vz = *idx_txz = *idx_tzz = *idx_txx = -1;

	for (k = 0; k < ncomp; k++) {
		switch (chan[k]) {
			case 1:
				verr("kh_misfit: component %d is trid=1 (P/hydrophone).  The "
				     "energy flux needs the traction components sig_xz and "
				     "sig_zz separately; P = 0.5*(Txx+Tzz) has already "
				     "discarded the deviatoric part that carries the shear "
				     "traction.  Use comp=_rvx,_rvz,_rtxz,_rtzz.", k);
				break;
			case 2: *idx_txz = k; break;
			case 3: *idx_tzz = k; break;
			case 4: *idx_txx = k; break;
			case 6: *idx_vx  = k; break;
			case 7: *idx_vz  = k; break;
			default:
				verr("kh_misfit: unexpected trid=%d on component %d.",
				     chan[k], k);
		}
	}

	if (*idx_vx < 0 || *idx_vz < 0 || *idx_txz < 0 || *idx_tzz < 0)
		verr("kh_misfit: the energy-flux objective needs Vx, Vz, Txz and Tzz "
		     "at every receiver (have Vx=%d Vz=%d Txz=%d Tzz=%d).  Use "
		     "comp=_rvx,_rvz,_rtxz,_rtzz.",
		     *idx_vx >= 0, *idx_vz >= 0, *idx_txz >= 0, *idx_tzz >= 0);
}


/*---------------------------------------------------------------------
 * khApply - one receiver, one time sample.
 *
 * r[]   in  : residual (syn - obs) for each component
 * psi[] out : adjoint source value for each component
 * returns   : the flux contribution w_r * P(r) to J
 *
 * Every component not named below is left at zero, which is correct for
 * Txx on a horizontal line and is also what keeps an unexpected extra
 * channel from quietly contributing.
 *-------------------------------------------------------------------*/
double khApply(int ncomp, float *psi, const float *r, const khPar *kh,
               int irec, int idx_vx, int idx_vz, int idx_txz, int idx_tzz,
               double *flux_shear, double *flux_normal)
{
	float  w, wn;
	double rs, rn;
	int    k;

	for (k = 0; k < ncomp; k++) psi[k] = 0.0f;

	w  = kh->w[irec];
	wn = w * kh->nz;

	/* r = n_z*(dsig_xz*dv_x + dsig_zz*dv_z), split into the shear and
	 * normal traction terms so the two can be reported separately. */
	rs = (double)kh->nz * (double)r[idx_txz] * (double)r[idx_vx];
	rn = (double)kh->nz * (double)r[idx_tzz] * (double)r[idx_vz];

	/* P(r) = r  =>  P'(r) = 1.  P' would otherwise multiply all four
	 * slots alike, so it rescales the source without disturbing the
	 * traction:velocity balance the cancellation depends on. */
	psi[idx_vx]  = wn * r[idx_txz];
	psi[idx_vz]  = wn * r[idx_tzz];
	psi[idx_tzz] = wn * r[idx_vz];
	psi[idx_txz] = wn * r[idx_vx];

	*flux_shear  += (double)w * rs;
	*flux_normal += (double)w * rn;

	return (double)w * (rs + rn);
}
