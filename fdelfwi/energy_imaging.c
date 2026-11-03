#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"fdelfwi.h"
#include"par.h"

/**
 * energy_imaging.c -- Energy-norm angle filter for the elastic FWI gradient.
 *
 * Elastic translation of the energy imaging condition of
 *
 *   Rocha, Tanushev & Sava, 2016, "Acoustic wavefield imaging using the energy
 *   norm", GEOPHYSICS 81(4), S151-S163.
 *
 * ---------------------------------------------------------------------------
 * 1. What the acoustic paper does
 * ---------------------------------------------------------------------------
 * Rocha measures the correlation of the source and receiver wavefields in the
 * energy norm of the wave equation rather than with a plain product.  For the
 * acoustic potential U the 4D operator and inner product are (eqs 11, 13, 16)
 *
 *     []U = ( v grad U , dU/dt ),      I_E = sum_t []U . []V
 *
 * For a pair of monochromatic plane waves with |k_s| = |k_r| = w/v and an
 * opening angle 2*theta between k_s and k_r, this collapses to (eqs 20-22)
 *
 *     []U~ . []V~* = [ -v^2 (k_s.k_r) + w^2 ] U~V~*  =  2 w^2 sin^2(theta) U~V~*
 *
 * i.e. the GRADIENT (potential) part carries -w^2 cos(2theta) and the TIME
 * DERIVATIVE (kinetic) part carries +w^2.  Flipping the sign of one of them
 * selects the complementary angle (eqs 26, 35, 36):
 *
 *     I_E(0 deg)  = sum_t ( dU/dt dV/dt + v grad U . v grad V) ~ +2w^2 sin^2
 *     I_E(90 deg) = sum_t (-dU/dt dV/dt + v grad U . v grad V) ~ -2w^2 cos^2
 *
 * sin^2(theta) peaks where the two wavefields are ANTI-parallel (2theta=180),
 * which is transmission -- the tomographic/"banana-doughnut" regime.
 * cos^2(theta) peaks where they are parallel (2theta=0), which is specular
 * reflection -- the migration regime.
 *
 * The empirical exponential sharpening of eq 45 is NOT used here; Rocha drops
 * it himself for Marmousi (p.S161) to avoid tuning alpha, reducing eq 45 to
 * eq 35.
 *
 * ---------------------------------------------------------------------------
 * 2. The elastic translation
 * ---------------------------------------------------------------------------
 * In a velocity-stress formulation the natural energy norm is the physical
 * elastic energy density, whose metric is
 *
 *     Q = diag( rho I , S ),      S = elastic compliance,
 *
 * the same Q that energy_weights.c uses on the data for data_weight=2.  The
 * energy inner product of two STATES is <u,u'>_E = rho v.v' + sig : S : sig'.
 *
 * The adjoint wavefield is NOT a state.  psi = -dJ/du is a DUAL vector, so
 * psi_v has units [J.s/m] and psi_sig units [J/Pa].  To put it in the energy
 * inner product it must first be mapped to a state through the inverse
 * metric, u_psi = Q^-1 psi, i.e. v_psi = psi_v/rho and sig_psi = C psi_sig.
 * The material factors then CANCEL:
 *
 *     <u_fwd, Q^-1 psi>_E = rho v.(psi_v/rho) + sig : S : (C psi_sig)
 *                         = v . psi_v  +  sig : psi_sig
 *
 * so the energy inner product of the forward state with the adjoint
 * multiplier is simply the duality pairing, and the two halves are
 *
 *     K(x) = sum_t dt ( vx_f psi_vx + vz_f psi_vz )             [kinetic ]
 *     P(x) = sum_t dt ( txx_f psi_txx + tzz_f psi_tzz
 *                                     + txz_f psi_txz )         [potential]
 *
 * with no material parameters anywhere, and no factor 2 on the shear term:
 * the 2 of sig_ij eps_ij is already absorbed by the 1/mu Voigt weight that
 * Q^-1 inverts.  Both halves are in [J], so they are commensurate and
 * equipartition holds, which is what makes the ratio below meaningful.
 * (Inserting rho and S explicitly instead -- treating psi as a state --
 * mis-scales the two halves by rho * rho*V^2 ~ 1.6e16, measured as
 * rms(P)/rms(K) = 1.2e-13 on the taylor_vs shot.)
 *
 * Unlike Rocha's acoustic potential, where dU/dt is a SCALAR and so carries no
 * angle factor at all, the elastic kinetic term contracts two VECTORS.  Both
 * halves therefore carry an angle dependence and the algebra has to be redone.
 *
 * 2a. The time-reversal sign on psi_v.  Write the system as
 *
 *     Q du/dt = A u + f,    A = [[0, div], [grad_sym, 0]],    A* = -A.
 *
 * Varying the Lagrangian gives the adjoint system Q dpsi/dt = A psi + dJ/du,
 * marched BACKWARD in time.  In the reversed variable tau = T - t that reads
 * Q dpsi/dtau = -A psi - (source), and the substitution sigma -> -sigma maps
 * -A back to +A.  So psi, read as a function of tau, is a physical state only
 * after ONE RELATIVE SIGN FLIP between its velocity and its stress:
 *
 *     ( psi_v , psi_sig )  ==  ( -v_a , +sig_a )   up to a common factor,
 *
 * with (v_a, sig_a) a genuine forward-propagating state in tau.  The stress
 * half is blind to this (sigma ~ n (x) n is even under n -> -n) but the
 * velocity half is not, and getting it wrong EXCHANGES the two filters.  This
 * is the term I first omitted; the depth-band diagnostic in test_energy_ic.sh
 * caught it.
 *
 * 2b. Derivation for a P-P pair.  Let c = cos(2theta) = n_s.n_a, and A_s, A_a
 * the velocity amplitudes.  A P plane wave has v = A n and
 * eps_ij = -(A/Vp) n_i n_j, so for two physical states
 *
 *     rho v_s.v_a       = (l+2m) A_s A_a c      / Vp^2
 *     sig_s : S : sig_a = [l + 2m c^2] A_s A_a  / Vp^2
 *
 * using rho = (l+2m)/Vp^2.  What this file accumulates is the pairing against
 * psi, so by 2a the kinetic half comes out NEGATED:
 *
 *     K = -(l + 2m) c,      P = l + 2m c^2
 *
 * dropping the common A_s A_a / Vp^2.  Hence
 *
 *     K + P = (1-c)(l - 2m c) =  2 sin^2(theta) * (l - 2m cos(2theta))
 *     K - P = -(1+c)(l + 2m c) = -2 cos^2(theta) * (l + 2m cos(2theta))
 *
 * so, with K and P as this file computes them,
 *
 *     K + P  is the TOMOGRAPHIC combination (vanishes at 2theta = 0  )
 *     K - P  is the MIGRATION   combination (vanishes at 2theta = 180)
 *
 * the SAME pairing as the acoustic potential formulation: Rocha's I_E(0 deg),
 * the sum of his two terms, is likewise the 2 sin^2 tomographic filter.
 *
 * The S-S pair behaves the same way.  With phi = 2theta, e the polarisation,
 * one finds rho v_s.v_a = cos(phi) and sig_s:S:sig_a = cos(2phi), so K = -cos
 * (phi), P = cos(2phi), and
 *
 *     K + P = (2cos(phi)+1)(cos(phi)-1)     vanishes at phi = 0
 *     K - P = -(2cos(phi)-1)(cos(phi)+1)    vanishes at phi = 180
 *
 * again tomographic and migration respectively.
 *
 * Two departures from the acoustic case follow from the algebra above:
 *
 *   (i)  the clean 2sin^2 / 2cos^2 factors are modulated by an extra isotropic
 *        factor (l -+ 2m cos2theta).  With Vp=2000, Vs=1150, rho=2000 this
 *        changes sign at 2theta = 59 deg (tomographic) and 121 deg
 *        (migration), so the filters are not strictly one-signed at
 *        intermediate angles.  They are still exact at 0 and 180 deg.
 *   (ii) P-S / S-P conversions do not separate at all: eq 21 needs a single
 *        |k| = w/v, and a converted pair has |k_s||k_r| = w^2/(Vp Vs).  The
 *        residue scales as (1 - Vp/Vs).
 *
 * Both (i) and (ii) were originally documented here as unavoidable -- "NOT
 * approximations introduced by this code".  That turned out to be wrong.  They
 * are artifacts of ONE member of a family of equally valid potential-energy
 * pairings, and they vanish identically in another.  See section 2c.
 *
 * ---------------------------------------------------------------------------
 * 2c. The gamma family, and why gamma = 0 removes (i) and (ii)
 * ---------------------------------------------------------------------------
 * The strain energy of a SINGLE isotropic field can be written in more than
 * one way.  The forms differ by the null Lagrangian
 *
 *     N(v,v~) = div.div~ - d_i v_j d_j v~_i
 *
 * whose integral over the domain is a pure boundary term, so every member
 * gives the same total energy.  They do NOT agree pointwise for the product of
 * two DIFFERENT fields, and an imaging condition is exactly such a product.
 * Parameterise the family by gamma:
 *
 *     V_gamma = (l+2m) div.div~ + m curl.curl~ - gamma * m * N(v,v~)
 *
 *     gamma = 2   eps:C:eps~         the strain-energy pairing  <- was hard-wired
 *     gamma = 1   Rocha's Laplacian form
 *     gamma = 0   (l+2m) div.div~ + m curl.curl~   (Navier / divergence-curl)
 *
 * With K at the matching (acceleration) level, the plane-wave algebra gives,
 * for an opening angle phi = 2theta and kappa = Vs^2/Vp^2,
 *
 *     PP:  K+V = 2sin^2(th) (1 - 2 gamma kappa cos^2(th))
 *          K-V = 2cos^2(th) (1 - 2 gamma kappa sin^2(th))
 *     SS:  the same with kappa -> 1
 *     PS:  V_gamma = -+ gamma sqrt(kappa) sin(phi) cos(phi),   K = +- sin(phi)
 *
 * At gamma = 0 all three collapse to something much better behaved:
 *
 *     PP and SS both give  K+V = 2sin^2(theta),  K-V = 2cos^2(theta)
 *         -- identical for the two modes, kappa has dropped out entirely, and
 *            there is no isotropic modulation, so no sign reversal anywhere.
 *            That is (i) gone.
 *     PS gives V_0 = 0 identically, hence c2 = |V-K|/(|V-K|+|V+K|) = 1/2
 *            EXACTLY at every angle, for every velocity ratio.
 *            That is (ii) turned from an uncontrolled leak into a constant.
 *
 * The gamma=2 zero crossing quoted in (i) is the same statement read the other
 * way: 1 - 2*2*kappa*cos^2(th) = 0 at cos^2(th) = 1/(4 kappa), which for
 * Vp=2000, Vs=1150 is 2theta = 59.2 deg.  It is a property of gamma, not of
 * elasticity.
 *
 * IMPORTANT, and the reason gamma=0 alone is not the whole answer: c2 = 1/2 on
 * a converted pair means the LINEAR weight a_tomo = |k+p|/(|k+p|+|k-p|) gives
 * conversions a weight of 0.5 -- half passed, not rejected.  Rejecting them
 * needs the sharpened mask exp(-alpha*c2) of Rocha eq 45, which this file
 * deliberately did not implement.  gamma=0 and energy_alpha>0 are therefore
 * complementary: gamma fixes the angle algebra, alpha turns it into a filter.
 *
 * 2d. gamma=0 is also immune to the 2a sign trap, but only half of it.
 * V_0 is built from psi_V (velocity gradients) while K is built from psi_SIGMA
 * (stress divergences), so the RELATIVE sign between the two halves is still
 * set by 2a and still has to be right.  What gamma=0 does remove is the mode
 * ambiguity: because the two filters no longer depend on kappa, a sign error
 * can no longer be mistaken for a legitimate mode-dependent modulation.  The
 * calibration used here is the abs_p cell of demo/gradient/test_rocha_fig9.sh,
 * where gamma=0 and gamma=2 must agree (both are exact at 2theta = 0 and 180):
 * if the direct-path banana lands in a_migr instead of a_tomo, the sign is
 * flipped.  Do not calibrate on anything at an intermediate angle.
 *
 * The alternative fix for (i) and (ii) suggested in earlier revisions of this
 * header -- taking cos(2theta) from the elastic power-flux (Poynting) vector
 * S_j = -sig_ij v_i -- is still valid and still exact, but it needs an extra
 * vector field per wavefield.  gamma=0 costs nothing beyond derivatives the
 * kernel already forms.  See 2e for why the Poynting route was NOT taken.
 * (NB: fdelmodc/decomposition.c is kx-w UP/DOWN separation of receiver data
 * after Wapenaar 1998 -- it is acoustic and does not do P/S on model-domain
 * wavefields.  Nothing in the tree currently does.)
 *
 * ---------------------------------------------------------------------------
 * 2e. EI_MODESEP -- Rocha's ACOUSTIC norm applied twice, once per mode
 * ---------------------------------------------------------------------------
 * gamma=0 removes the isotropic modulation and pins converted pairs at a
 * single known c2, but it still measures the angle through ONE elastic
 * pairing.  There is a cheaper and exact alternative, which is simply to
 * notice that the two Helmholtz scalars each satisfy Rocha's own equation.
 * In a locally homogeneous isotropic medium, taking div and curl of
 * rho vdot.. = (l+m) grad(div v) + m lap v gives
 *
 *     D = div  v   ->   rho Ddot. = (l+2m) lap D    i.e. a scalar wave at Vp
 *     C = curl v   ->   rho Cdot. =     m  lap C    i.e. a scalar wave at Vs
 *
 * so D and C are each a legitimate acoustic "U" in the sense of eq 1, with
 * their OWN velocity.  Rocha's 4-gradient and inner product (eqs 13, 16) then
 * apply verbatim, once per mode:
 *
 *     K_P = sum_t dt  Ddot_f Ddot_a        V_P = sum_t dt Vp^2 grad D_f . grad D_a
 *     K_S = sum_t dt  Cdot_f Cdot_a        V_S = sum_t dt Vs^2 grad C_f . grad C_a
 *
 * PLANE-WAVE RESULT, which is the whole point.  For D = D0 cos(k.x - wt),
 *
 *     Ddot  = w D0 sin,        grad D = -k D0 sin,
 *     K_P ~ w_f w_a D_f D_a,   V_P ~ Vp^2 (k_f . k_a) D_f D_a,
 *
 * and with |k| = w/Vp for BOTH factors -- true here by construction, because
 * D only ever carries P energy --
 *
 *     K_P +- V_P  ~  w^2 (1 +- cos psi) D_f D_a,   psi = angle(k_f, k_a),
 *
 * hence a_tomo = cos^2(psi/2), a_migr = sin^2(psi/2).  The SAME expression
 * comes out of the S channel with Vs.  No velocity ratio kappa, no isotropic
 * (l -+ 2m cos2theta) factor, no sign reversal at intermediate angles.  Items
 * (i) of section 2b and the kappa dependence of 2c are gone EXACTLY, not
 * asymptotically, and they are gone for both modes with one expression.
 *
 * psi is measured in the COMMON LOOP-TIME frame, i.e. between the two arrays
 * as the backward loop holds them, not between the two physical propagation
 * directions.  On the direct path the adjoint field, read as a function of
 * increasing loop time t, converges on the receiver -- the same direction the
 * forward field travels -- so transmission is psi = 0 and a_tomo = 1 there,
 * which is why the sum is again the tomographic combination.  This is the
 * same frame convention as Rocha eq 16 ("the receiver wavefield is oriented
 * backward in time").
 *
 * WHY NOT POYNTING.  The elastic power flux S = -sig.v measures the two
 * propagation directions separately and so would also remove (i) and (ii),
 * and it would additionally give the bisector (reflector dip).  It was not
 * used because an imaging condition is BILINEAR -- one factor of each field --
 * while S is QUADRATIC IN A SINGLE FIELD.  Three consequences, none of which
 * the formulation above has:
 *   - |S| ~ cos^2(phase), so the normalisation S/|S| passes through zero
 *     twice per period;
 *   - two arrivals at one point give S = S_11 + S_12 + S_21 + S_22, and the
 *     cross terms point nowhere physical.  Suppressing them needs a running
 *     time average whose window is set by the beat period between arrivals,
 *     which is not a controlled quantity;
 *   - S mixes modes: S = S_PP + S_PS + S_SP + S_SS at any point carrying both.
 * K and V above are bilinear, so the angle factor comes out multiplied by the
 * same envelope D_f D_a and none of the three arise.  The Poynting route
 * remains the way to get the bisector if angle GATHERS are ever wanted.
 *
 * 2e-i. What is actually differenced.  Ddot is not stored or time-differenced;
 * it is read off the OTHER half of the system,
 *
 *     Ddot = d/dt (div v) = div (dv/dt) = div ( b grad.sig ) = div a,
 *
 * so the accelerations that the velocity kernel forms anyway are divided and
 * curled with the SAME two helpers that produce D and C.  No time history, no
 * extra time level, and Ddot lands on the P node and Cdot on the Txz node --
 * exactly where D and C live.
 *
 * 2e-ii. Sign.  Both K halves are built from psi_SIGMA and both V halves from
 * psi_V, through the IDENTICAL mappings gamma=0 uses (ei_sa* and ei_va*).  The
 * relative sign between K and V is therefore inherited from the gamma=0
 * calibration of section 2d and is not re-derived here.  The check is the same
 * one: on the abs_p cell of demo/gradient/test_rocha_fig9.sh the direct-path
 * banana must land in a_tomo, and EI_MODESEP must agree with gamma=0 there,
 * since both are exact at psi = 0 and 180.
 *
 * 2e-iii. Blind spot, stated plainly.  div annihilates S and curl annihilates
 * P, so a CONVERTED pair (forward P with adjoint S, or the reverse) gives
 * D_a = 0 and C_f = 0 and therefore contributes to NEITHER channel.  The mask
 * at such a point is set entirely by whatever unconverted energy is also
 * present; where there is none, the denominator falls to the eps floor and the
 * weight goes to zero.  This is a different failure from gamma=0, where a
 * converted pair returns c2 = 1/2 and is HALF-PASSED by the linear weight.
 * Neither measures the conversion angle.  Only a Poynting estimate would, and
 * see above for the price.
 *
 * 2e-iv. Noise.  V is a second spatial derivative of the velocities and K a
 * second spatial derivative of the stresses, one order higher than at gamma=0
 * or 2, so both halves amplify near-Nyquist content by |k|^2.  They amplify it
 * EQUALLY, so the ratio that forms the weight is to first order unaffected;
 * what is affected is the eps floor, which is set from the mean denominator
 * and will sit higher.  Worth watching if the mask looks speckled.
 *
 * ---------------------------------------------------------------------------
 * 3. Normalised weight, and why it is normalised this way
 * ---------------------------------------------------------------------------
 * Rocha's I_E is an IMAGE, not a weight: he replaces the cross-correlation
 * outright.  Our gradient is not one correlation but three different ones
 * (lambda, mu, rho), so replacing them would mean rebuilding every parameter
 * kernel inside the energy inner product (Rocha eq 42).  Instead this file
 * applies the energy inner product as a dimensionless, per-point ANGLE WEIGHT
 * on the conventional kernels:
 *
 *     a_tomo(x) = |k+p| / ( |k+p| + |k-p| + eps )
 *     a_migr(x) = |k-p| / ( |k+p| + |k-p| + eps )
 *
 * Properties:
 *   - for a single plane-wave pair the denominator is 2w^2 * (modulation), so
 *     a_tomo -> sin^2(theta) and a_migr -> cos^2(theta) up to item (i) above;
 *   - both lie in [0,1] and a_tomo + a_migr = 1 exactly (up to eps), so
 *     g*a_tomo + g*a_migr = g.  That identity is a usable regression check;
 *   - the w^2 that Rocha has to compensate for by integrating the source and
 *     the data (his eqs 28-31) CANCELS in the ratio, so no compensation is
 *     needed here;
 *   - no empirical parameter.
 *
 * THIS WEIGHT FORM IS AN APPROXIMATION to Rocha eq 42: it filters the
 * conventional kernels by scattering angle instead of rederiving them in the
 * energy metric.  The exact form is the follow-up.
 *
 * K and P are combined AS ACCUMULATED -- they are NOT first rescaled to equal
 * global RMS.  That rescaling was applied here historically, on the argument
 * that psi_v and psi_sigma are Lagrange multipliers whose relative
 * normalisation is set by the discrete adjoint rather than by physics, so a
 * global scalar per shot should restore kinetic/potential equipartition.  It
 * does the opposite.  The weight must return 1 exactly where the two halves
 * are EQUAL; multiplying them by different constants sK != sP destroys that
 * condition and manufactures a floor c2 = |r-1|/(|r+1|+|r-1|), r = sK/sP,
 * everywhere -- 0.12 at the measured r = 0.77, which through exp(-alpha c2)
 * dims the entire transmission kernel by ~2x.  The halves ARE commensurate in
 * absolute terms (see the SCALING notes on the gamma=0 accumulators and
 * test_ei_planewave.c, which checks both against rho A^2 w^2 dt sin^2(phi)
 * pointwise), so there is nothing legitimate for a global scalar to fix.
 *
 * rms(P)/rms(K) is still measured and reported at verbose>0, now purely as a
 * diagnostic: it is the cheap check that the two accumulators are commensurate
 * at all.  Dividing it out made exactly the bugs it would have caught
 * invisible.
 *
 * Like grad_taper, res_taper and smooth_grad, a non-zero energy_ic changes the
 * gradient and therefore BREAKS the Taylor test by construction.
 *
 * AUTHOR:
 *   fdelfwi -- elastic FWI
 */

#define MAX(x,y) ((x) > (y) ? (x) : (y))

/* Floor on the weight denominator, as a fraction of its MEAN over the imaged
 * interior.  A fraction of the maximum will not do: the denominator peaks by
 * orders of magnitude at the source, so a 1e-3 fraction of that still swamps
 * the denominator everywhere else and visibly breaks a_tomo + a_migr = 1. */
#define EI_EPS_FRAC 1.0e-4f

static int    ei_mode  = EI_OFF;
static int    ei_gamma = EI_GAMMA2;
static float  ei_alpha = 0.0f;   /* 0 = linear weight; >0 = exp(-alpha*c2) */
static size_t ei_sizem = 0;
static float *ei_K     = NULL;   /* kinetic   accumulation, P grid */
static float *ei_P     = NULL;   /* potential accumulation, P grid */
static float *ei_w     = NULL;   /* applied weight, P grid         */

/* Scratch for gamma=0 and EI_MODESEP.  sa* hold the adjoint STATE stress
 * C*psi_sigma; va* hold the adjoint STATE velocity psi_v/rho, including the
 * dx/dt that makes it commensurate with the kinetic half.  NULL at gamma=2.
 * EI_MODESEP uses the SAME two mappings, which is what makes its K/V sign
 * inherit the gamma=0 calibration (section 2e-ii). */
static float *ei_saxx = NULL, *ei_sazz = NULL, *ei_saxz = NULL;
static float *ei_vax  = NULL, *ei_vaz  = NULL;

/* EI_MODESEP only.  ei_K and ei_P are reused as the P-channel pair (K_P,V_P);
 * these two add the S channel.  Keeping the P channel in the original arrays
 * is what lets EI_GET_KIN/EI_GET_POT keep working for callers that predate
 * the mode split. */
static float *ei_KS = NULL, *ei_VS = NULL;

/* EI_MODESEP scratch, all on their native staggered nodes:
 *   af*, aa* : forward and adjoint acceleration, Vx and Vz nodes
 *   Df, Da   : div v,  P node        Cf, Ca : curl v, Txz node
 * Df..Ca must be materialised rather than computed inline because their
 * spatial GRADIENT is what the potential half needs. */
static float *ei_afx = NULL, *ei_afz = NULL;
static float *ei_aax = NULL, *ei_aaz = NULL;
static float *ei_Df  = NULL, *ei_Da  = NULL;
static float *ei_Cf  = NULL, *ei_Ca  = NULL;


/* Fill c1..c4 with the staggered first-derivative coefficients for this
 * spatial order.  Same table as fwi_gradient.c. */
static void ei_fdcoef(int iorder, float *c1, float *c2, float *c3, float *c4)
{
	*c1 = 9.0f/8.0f;   *c2 = -1.0f/24.0f;   *c3 = 0.0f;   *c4 = 0.0f;
	switch (iorder) {
		case 6:
			*c1 = 75.0f/64.0f;    *c2 = -25.0f/384.0f;
			*c3 = 3.0f/640.0f;    break;
		case 8:
			*c1 = 1225.0f/1024.0f; *c2 = -245.0f/3072.0f;
			*c3 = 49.0f/5120.0f;   *c4 = -5.0f/7168.0f;  break;
		default: break;          /* 2nd and 4th both use the 4th-order pair */
	}
	if (iorder == 2) { *c1 = 1.0f; *c2 = 0.0f; }
}


/***********************************************************************
 * ei_bounds -- Loop bounds matching accumGradient (fwi_gradient.c).
 *
 * Points excluded here are exactly the points where accumGradient writes
 * nothing, so the weight there is irrelevant.
 ***********************************************************************/
static void ei_bounds(modPar *mod, bndPar *bnd,
                      int *ibPx, int *iePx, int *ibPz, int *iePz,
                      int *ibTx, int *ieTx, int *ibTz, int *ieTz)
{
	int half = mod->iorder / 2;
	int nax = mod->nax, n1 = mod->naz;

	*ibPx = mod->ioPx; *iePx = mod->iePx;
	*ibPz = mod->ioPz; *iePz = mod->iePz;
	if (bnd->lef == 4 || bnd->lef == 2) *ibPx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) *iePx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) *ibPz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) *iePz -= bnd->ntap;
	if (bnd->top == 1) *ibPz = MAX(*ibPz, mod->ioPz + 1);
	*ibPx = MAX(*ibPx, half - 1);
	*iePx = MAX(*ibPx, *iePx);
	if (*iePx > nax - half) *iePx = nax - half;
	*ibPz = MAX(*ibPz, half - 1);
	*iePz = MAX(*ibPz, *iePz);
	if (*iePz > n1 - half) *iePz = n1 - half;

	*ibTx = mod->ioTx; *ieTx = mod->ieTx;
	*ibTz = mod->ioTz; *ieTz = mod->ieTz;
	if (bnd->lef == 4 || bnd->lef == 2) *ibTx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) *ieTx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) *ibTz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) *ieTz -= bnd->ntap;
	if (bnd->top == 1) *ibTz = MAX(*ibTz, mod->ioTz + 1);
	*ibTx = MAX(*ibTx, half);
	*ieTx = MAX(*ibTx, *ieTx);
	if (*ieTx > nax - half) *ieTx = nax - half;
	*ibTz = MAX(*ibTz, half);
	*ieTz = MAX(*ibTz, *ieTz);
	if (*ieTz > n1 - half) *ieTz = n1 - half;
}


/***********************************************************************
 * energyImagingInit -- Allocate and zero the two accumulators.
 *
 * Must be called once per shot, BEFORE adj_shot.  mode = EI_OFF frees
 * everything and turns the accumulators into no-ops.
 *
 * No material arrays are needed: the Q^-1 map that turns the adjoint
 * multiplier into a state cancels rho and the compliance exactly (see the
 * derivation at the top of this file).
 ***********************************************************************/
void energyImagingInit(modPar *mod, int mode, int gamma, float alpha,
                       int verbose)
{
	size_t sizem;

	energyImagingFree();
	ei_mode  = mode;
	ei_gamma = gamma;
	ei_alpha = alpha;
	if (mode == EI_OFF) return;

	if (mod->ischeme <= 2)
		verr("energyImagingInit: energy_ic=%d needs the elastic scheme "
		     "(ischeme=3); got ischeme=%d.", mode, mod->ischeme);
	if (gamma != EI_GAMMA0 && gamma != EI_GAMMA2 && gamma != EI_MODESEP)
		verr("energyImagingInit: energy_gamma=%d not implemented; use 2 "
		     "(eps:C:eps), 0 (divergence-curl) or -1 (per-mode acoustic).",
		     gamma);
	if (alpha < 0.0f)
		verr("energyImagingInit: energy_alpha=%g must be >= 0.", alpha);

	sizem   = (size_t)mod->nax * mod->naz;
	ei_sizem = sizem;
	ei_K   = (float *)calloc(sizem, sizeof(float));
	ei_P   = (float *)calloc(sizem, sizeof(float));
	ei_w   = (float *)calloc(sizem, sizeof(float));
	if (!ei_K || !ei_P || !ei_w)
		verr("energyImagingInit: allocation failed for %zu points.", sizem);

	if (gamma == EI_GAMMA0 || gamma == EI_MODESEP) {
		ei_saxx = (float *)calloc(sizem, sizeof(float));
		ei_sazz = (float *)calloc(sizem, sizeof(float));
		ei_saxz = (float *)calloc(sizem, sizeof(float));
		ei_vax  = (float *)calloc(sizem, sizeof(float));
		ei_vaz  = (float *)calloc(sizem, sizeof(float));
		if (!ei_saxx || !ei_sazz || !ei_saxz || !ei_vax || !ei_vaz)
			verr("energyImagingInit: adjoint-state scratch allocation failed "
			     "for %zu points.", sizem);
	}

	if (gamma == EI_MODESEP) {
		ei_KS  = (float *)calloc(sizem, sizeof(float));
		ei_VS  = (float *)calloc(sizem, sizeof(float));
		ei_afx = (float *)calloc(sizem, sizeof(float));
		ei_afz = (float *)calloc(sizem, sizeof(float));
		ei_aax = (float *)calloc(sizem, sizeof(float));
		ei_aaz = (float *)calloc(sizem, sizeof(float));
		ei_Df  = (float *)calloc(sizem, sizeof(float));
		ei_Da  = (float *)calloc(sizem, sizeof(float));
		ei_Cf  = (float *)calloc(sizem, sizeof(float));
		ei_Ca  = (float *)calloc(sizem, sizeof(float));
		if (!ei_KS || !ei_VS || !ei_afx || !ei_afz || !ei_aax || !ei_aaz ||
		    !ei_Df || !ei_Da || !ei_Cf || !ei_Ca)
			verr("energyImagingInit: EI_MODESEP scratch allocation failed "
			     "for %zu points.", sizem);
	}

	if (verbose) {
		const char *what = (mode == EI_TOMO) ? "tomographic (|K+P|, keeps transmission)"
		                 : (mode == EI_MIGR) ? "migration (|K-P|, keeps reflections)"
		                 : "diagnostics only (gradient untouched)";
		vmess("Energy imaging condition (energy_ic=%d): %s", mode, what);
		vmess("  energy_gamma=%d: %s", gamma,
		      (gamma == EI_MODESEP)
		        ? "per-mode acoustic norm (div at Vp, curl at Vs) -- exact "
		          "(1 +- cos psi), no kappa, blind to P-S"
		      : (gamma == EI_GAMMA0)
		        ? "(l+2m)div.div~ + m curl.curl~  -- mode independent, V=0 on P-S"
		        : "eps:C:eps~  -- modulated by (l -+ 2m cos2theta), P-S leaks");
		if (gamma == EI_MODESEP)
			vmess("  energy_gamma=-1: g_lam gets the P-channel weight (it is a "
			      "pure-PP kernel); g_mu and g_rho get the energy-weighted mix.");
		if (alpha > 0.0f)
			vmess("  energy_alpha=%g: sharpened mask exp(-alpha*c2) (Rocha eq 45). "
			      "NOTE a_tomo + a_migr != 1 with alpha>0.", alpha);
		else
			vmess("  energy_alpha=0: linear weight, a_tomo + a_migr = 1.");
	}
}


/***********************************************************************
 * Staggered divergence of a stress tensor, raw stencil (no 1/dx).
 *
 * _x is evaluated at the Vx node, _z at the Vz node -- the same stencils as
 * accumGradient_rho_Dsig in fwi_gradient.c and Phase F1 of the elastic
 * kernel, so (grad.tau)_x here is exactly the quantity the forward velocity
 * update consumes.
 ***********************************************************************/
static float ei_divtau_x(float *txx, float *txz, int ig, int n1,
                         int iorder, float c1, float c2, float c3, float c4)
{
	float d = c1*(txx[ig]      - txx[ig-n1]   + txz[ig+1] - txz[ig])
	        + c2*(txx[ig+n1]   - txx[ig-2*n1] + txz[ig+2] - txz[ig-1]);
	if (iorder >= 6)
		d += c3*(txx[ig+2*n1] - txx[ig-3*n1] + txz[ig+3] - txz[ig-2]);
	if (iorder >= 8)
		d += c4*(txx[ig+3*n1] - txx[ig-4*n1] + txz[ig+4] - txz[ig-3]);
	return d;
}

static float ei_divtau_z(float *tzz, float *txz, int ig, int n1,
                         int iorder, float c1, float c2, float c3, float c4)
{
	float d = c1*(tzz[ig]   - tzz[ig-1] + txz[ig+n1]   - txz[ig])
	        + c2*(tzz[ig+1] - tzz[ig-2] + txz[ig+2*n1] - txz[ig-n1]);
	if (iorder >= 6)
		d += c3*(tzz[ig+2] - tzz[ig-3] + txz[ig+3*n1] - txz[ig-2*n1]);
	if (iorder >= 8)
		d += c4*(tzz[ig+3] - tzz[ig-4] + txz[ig+4*n1] - txz[ig-3*n1]);
	return d;
}


/***********************************************************************
 * ei_div_v / ei_curl_v -- velocity divergence at the P node and curl at the
 * Txz node, to the scheme's own order.
 *
 * Stencils and order branches are taken verbatim from the lambda/mu loops of
 * fwi_gradient.c, so the div that V_0 pairs is the SAME discrete operator the
 * lambda kernel uses.  Both are needed for the forward field and for the
 * adjoint field, hence helpers rather than four inlined copies.
 *
 * curl_y = dvx/dz - dvz/dx.  The antisymmetric part of grad v does no work, so
 * unlike div it cannot be recovered from the stresses -- this is the one
 * quantity gamma=0 needs that gamma=2 does not.
 ***********************************************************************/
static float ei_div_v(float *vx, float *vz, int ix, int iz, int n1,
                      float sdx, float sdz, int io,
                      float c1, float c2, float c3, float c4)
{
	int ig = ix*n1+iz;
	float dx = sdx*(c1*(vx[(ix+1)*n1+iz] - vx[ig])
	               +c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]));
	float dz = sdz*(c1*(vz[ig+1] - vz[ig])
	               +c2*(vz[ig+2] - vz[ig-1]));
	if (io >= 6) {
		dx += sdx*c3*(vx[(ix+3)*n1+iz] - vx[(ix-2)*n1+iz]);
		dz += sdz*c3*(vz[ig+3] - vz[ig-2]);
	}
	if (io >= 8) {
		dx += sdx*c4*(vx[(ix+4)*n1+iz] - vx[(ix-3)*n1+iz]);
		dz += sdz*c4*(vz[ig+4] - vz[ig-3]);
	}
	return dx + dz;
}

static float ei_curl_v(float *vx, float *vz, int ix, int iz, int n1,
                       float sdx, float sdz, int io,
                       float c1, float c2, float c3, float c4)
{
	int ig = ix*n1+iz;
	float dxdz = sdz*(c1*(vx[ig]   - vx[ig-1])
	                 +c2*(vx[ig+1] - vx[ig-2]));
	float dzdx = sdx*(c1*(vz[ig]             - vz[(ix-1)*n1+iz])
	                 +c2*(vz[(ix+1)*n1+iz]   - vz[(ix-2)*n1+iz]));
	if (io >= 6) {
		dxdz += sdz*c3*(vx[ig+2] - vx[ig-3]);
		dzdx += sdx*c3*(vz[(ix+2)*n1+iz] - vz[(ix-3)*n1+iz]);
	}
	if (io >= 8) {
		dxdz += sdz*c4*(vx[ig+3] - vx[ig-4]);
		dzdx += sdx*c4*(vz[(ix+3)*n1+iz] - vz[(ix-4)*n1+iz]);
	}
	return dxdz - dzdx;
}


/***********************************************************************
 * ei_accum_kinetic_gamma0 -- K = b (grad.tau_f).(grad.sigma_a), one step.
 *
 * The gamma=0 kinetic half, at ACCELERATION level so that it is commensurate
 * with V_0, which is built from velocity GRADIENTS (section 2c).  Using the
 * velocity-level pairing v_f.psi_v here instead would leave K a factor w^2
 * below V_0, which destroys the exact zero at transmission -- the whole point
 * of the filter.  Nothing downstream rescales the halves, so an error of this
 * kind is not absorbed: it shows up directly in rms(P)/rms(K).
 *
 * Called where psi_sigma is valid.  Two mappings are needed:
 *   - the adjoint STATE stress is sigma_a = C psi_sigma (the Q^-1 map).  There
 *     is no cancellation here as there is at gamma=2, because V_0 is not the
 *     Q inner product, so C is applied explicitly.
 *   - rho d(v)/dt = grad.sigma, so grad.tau IS the acceleration up to rho.
 *
 * fdelmodc stores T_ij = -sigma_ij.  Both factors carry that sign and it
 * cancels in the product, exactly as for the stress-stress pairing at gamma=2.
 *
 * SCALING.  It is not enough for each half to be right up to "some global
 * constant" -- the two halves are added and subtracted, so only their RATIO
 * matters, and a constant that differs between them is a real error: it moves
 * the zero of |K-P| away from transmission, which is the whole point.
 * ei_build_weight no longer rescales the halves, so such an error is not
 * hidden -- it appears directly in the rms(P)/rms(K) report.
 * rox = dt*b/dx, and ei_divtau_* returns RAW differences, so each of the two
 * still needs its own 1/dx.  The physical per-step contribution is therefore
 *     dt * (rox*dx/dt) * sdx**2 * D_f*D_a  =  rox*sdx * D_f*D_a,
 * i.e. the dt of the time integral cancels exactly against the dt inside rox.
 * It is cancelled, not dropped, so dt does not appear in the loops below.
 *
 * Like the fdelmodc velocity kernel itself -- which multiplies both the x- and
 * the z-difference by the single factor rox -- this assumes dx == dz.
 ***********************************************************************/
static void ei_accum_kinetic_gamma0(modPar *mod,
                                    float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                                    wflPar *wfl_adj, float dt)
{
	int ix, iz, n1 = mod->naz, io = mod->iorder;
	size_t i;
	float c1, c2, c3, c4;
	/* 0.5 is the staggered split over the two cells either side of the node. */
	float kx = 0.5f/mod->dx, kz = 0.5f/mod->dz;

	ei_fdcoef(io, &c1, &c2, &c3, &c4);

	/* sigma_a = C psi_sigma.  l2m/lam live at the P node, muu at the Txz
	 * node, matching where psi_txx/psi_tzz and psi_txz are defined. */
	for (i = 0; i < ei_sizem; i++) {
		float pxx = wfl_adj->txx[i], pzz = wfl_adj->tzz[i];
		ei_saxx[i] = mod->l2m[i]*pxx + mod->lam[i]*pzz;
		ei_sazz[i] = mod->lam[i]*pxx + mod->l2m[i]*pzz;
		ei_saxz[i] = mod->muu[i]*wfl_adj->txz[i];
	}

	for (ix = mod->ioXx; ix < mod->ieXx; ix++) {
		for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
			int ig = ix*n1+iz;
			float Df = ei_divtau_x(fwd_txx, fwd_txz, ig, n1, io, c1, c2, c3, c4);
			float Da = ei_divtau_x(ei_saxx, ei_saxz, ig, n1, io, c1, c2, c3, c4);
			float prod = kx * mod->rox[ig] * Df * Da;
			ei_K[ig-n1] += prod;
			ei_K[ig]    += prod;
		}
	}
	for (ix = mod->ioZx; ix < mod->ieZx; ix++) {
		for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
			int ig = ix*n1+iz;
			float Df = ei_divtau_z(fwd_tzz, fwd_txz, ig, n1, io, c1, c2, c3, c4);
			float Da = ei_divtau_z(ei_sazz, ei_saxz, ig, n1, io, c1, c2, c3, c4);
			float prod = kz * mod->roz[ig] * Df * Da;
			ei_K[ig-1] += prod;
			ei_K[ig]   += prod;
		}
	}
}


/***********************************************************************
 * ei_accum_potential_gamma0 -- V_0 = (l+2m) div.div~ + m curl.curl~.
 *
 * Called where psi_v is valid.  The adjoint STATE velocity is b*psi_v, and
 * rox/roz are dt*b/dx, so b is recovered exactly as rox*dx/dt.  The dx/dt is
 * NOT optional: dropping it leaves this half a factor dx**2 below the kinetic
 * one (see the SCALING note on ei_accum_kinetic_gamma0).  The 1/rho has to sit
 * INSIDE the derivative, which is why it is applied to a scratch copy before
 * differentiating rather than pulled out as a factor.
 *
 * div is native at the P node, curl at the Txz node (the antisymmetric part of
 * grad v does no work and therefore cannot be recovered from the stresses --
 * this is the one quantity gamma=0 needs that gamma=2 does not).  The curl
 * term is scattered to the 4 surrounding P points with weight 0.25, the same
 * pattern the shear term already uses.
 ***********************************************************************/
static void ei_accum_potential_gamma0(modPar *mod, bndPar *bnd,
                                      float *fwd_vx, float *fwd_vz,
                                      wflPar *wfl_adj, float dt)
{
	int ix, iz, n1 = mod->naz, io = mod->iorder;
	int ibPx, iePx, ibPz, iePz, ibTx, ieTx, ibTz, ieTz;
	size_t i;
	float c1, c2, c3, c4, sdx = 1.0f/mod->dx, sdz = 1.0f/mod->dz;
	float bscx, bscz;

	ei_fdcoef(io, &c1, &c2, &c3, &c4);
	ei_bounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz, &ibTx, &ieTx, &ibTz, &ieTz);

	/* b = rox*dx/dt at the Vx node, b = roz*dz/dt at the Vz node. */
	bscx = mod->dx/dt;
	bscz = mod->dz/dt;
	for (i = 0; i < ei_sizem; i++) {
		ei_vax[i] = wfl_adj->vx[i] * mod->rox[i] * bscx;
		ei_vaz[i] = wfl_adj->vz[i] * mod->roz[i] * bscz;
	}

	/* Dilatational part, native at the P node. */
	for (ix = ibPx; ix < iePx; ix++) {
		for (iz = ibPz; iz < iePz; iz++) {
			int ig = ix*n1+iz;
			float df = ei_div_v(fwd_vx, fwd_vz, ix, iz, n1, sdx, sdz, io,
			                    c1, c2, c3, c4);
			float da = ei_div_v(ei_vax, ei_vaz, ix, iz, n1, sdx, sdz, io,
			                    c1, c2, c3, c4);

			ei_P[ig] += dt * mod->l2m[ig] * df * da;
		}
	}

	/* Rotational part, native at the Txz node. */
	for (ix = ibTx; ix < ieTx; ix++) {
		for (iz = ibTz; iz < ieTz; iz++) {
			int ig = ix*n1+iz;
			float cf = ei_curl_v(fwd_vx, fwd_vz, ix, iz, n1, sdx, sdz, io,
			                     c1, c2, c3, c4);
			float ca = ei_curl_v(ei_vax, ei_vaz, ix, iz, n1, sdx, sdz, io,
			                     c1, c2, c3, c4);
			float shear;

			shear = 0.25f * dt * mod->muu[ig] * cf * ca;
			ei_P[(ix-1)*n1+iz-1] += shear;
			ei_P[ix*n1+iz-1]     += shear;
			ei_P[(ix-1)*n1+iz]   += shear;
			ei_P[ig]             += shear;
		}
	}
}


/***********************************************************************
 * ei_ms_bounds -- the single interior box EI_MODESEP works on.
 *
 * EI_MODESEP needs ONE more derivative than gamma=0 or 2: D and C are first
 * derivatives of the wavefield, and the potential half differentiates them
 * again.  So there are two nested boxes.  wide = where D, C, and the
 * accelerations are FILLED; narrow = wide shrunk by half, where they are READ
 * through a stencil of reach `half`.  Reading at the narrow edge therefore
 * touches exactly the wide edge and no further.
 *
 * Both are derived from ei_bounds, so the taper, free-surface and stencil
 * exclusions are inherited; EI_MODESEP simply images a box `half` cells
 * narrower than the other gammas.  Everything outside stays at the calloc
 * zero and is never written, which is what makes the stencil reads safe
 * without a per-step memset.
 *
 * P-grid and Txz-grid bounds are intersected into one box because the two
 * channels scatter onto each other's nodes.
 ***********************************************************************/
static void ei_ms_bounds(modPar *mod, bndPar *bnd, int wide,
                         int *ibx, int *iex, int *ibz, int *iez)
{
	int ibPx, iePx, ibPz, iePz, ibTx, ieTx, ibTz, ieTz;
	int half = mod->iorder / 2;
	int m    = wide ? 0 : half;

	ei_bounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz, &ibTx, &ieTx, &ibTz, &ieTz);

	*ibx = MAX(ibPx, ibTx) + m;
	*ibz = MAX(ibPz, ibTz) + m;
	*iex = ((iePx < ieTx) ? iePx : ieTx) - m;
	*iez = ((iePz < ieTz) ? iePz : ieTz) - m;

	/* The gradient helpers below reach one cell further than ei_div_v does
	 * (ei_dp touches ig+4*st at iorder=8), hence half+1 rather than half. */
	if (*ibx < half + 1) *ibx = half + 1;
	if (*ibz < half + 1) *ibz = half + 1;
	if (*iex > mod->nax - half - 1) *iex = mod->nax - half - 1;
	if (*iez > mod->naz - half - 1) *iez = mod->naz - half - 1;
	if (*iex < *ibx) *iex = *ibx;
	if (*iez < *ibz) *iez = *ibz;
}


/***********************************************************************
 * ei_dm / ei_dp -- staggered first difference of a SCALAR field, raw
 * (no 1/dx), stride-generic so one pair covers x and z.
 *
 * These are the same two operators the kernel uses on vector components,
 * applied here to D and C.  The registration is what matters:
 *
 *   ei_dm on a P-node scalar  -> Vx node (stride n1) or Vz node (stride 1)
 *   ei_dp on a Txz-node scalar-> Vz node (stride n1) or Vx node (stride 1)
 *
 * so grad D and grad C both land on the velocity nodes, where the material
 * factors Vp^2 and Vs^2 can be formed from rox/roz, and from there each
 * product is scattered onto the P grid with weight 1/2 -- the same pattern
 * grad_rho uses in fwi_gradient.c.
 ***********************************************************************/
static float ei_dm(float *s, int ig, int st, int io,
                   float c1, float c2, float c3, float c4)
{
	float d = c1*(s[ig] - s[ig-st]) + c2*(s[ig+st] - s[ig-2*st]);
	if (io >= 6) d += c3*(s[ig+2*st] - s[ig-3*st]);
	if (io >= 8) d += c4*(s[ig+3*st] - s[ig-4*st]);
	return d;
}

static float ei_dp(float *s, int ig, int st, int io,
                   float c1, float c2, float c3, float c4)
{
	float d = c1*(s[ig+st] - s[ig]) + c2*(s[ig+2*st] - s[ig-st]);
	if (io >= 6) d += c3*(s[ig+3*st] - s[ig-2*st]);
	if (io >= 8) d += c4*(s[ig+4*st] - s[ig-3*st]);
	return d;
}


/***********************************************************************
 * ei_accum_modesep_at_sigma -- the two KINETIC halves, K_P and K_S.
 *
 * Ddot and Cdot are not time-differenced.  Using the momentum equation,
 *
 *     Ddot = d/dt div v = div (dv/dt) = div a,     Cdot = curl a,
 *
 * so this reduces to forming the two accelerations and then running the
 * SAME div/curl helpers over them that the potential half runs over the
 * velocities.  a = b grad.sig; ei_divtau_* returns RAW differences and
 * rox = dt*b/dx, so a = (rox/dt) * divtau_raw exactly -- the 1/dx is already
 * inside rox and the dt is divided back out.
 *
 * fdelmodc stores T_ij = -sigma_ij.  Both accelerations are built through
 * ei_divtau_*, so both carry that sign and it cancels in the product, the
 * same way it does for the stress-stress pairing at gamma=2.
 *
 * The adjoint STATE stress sigma_a = C psi_sigma uses the IDENTICAL mapping
 * as gamma=0 (ei_sa*), which is what makes the K/V relative sign inherit the
 * gamma=0 calibration rather than needing its own (section 2e-ii).
 *
 * Ddot lands on the P node and is accumulated there directly.  Cdot lands on
 * the Txz node and is scattered to its 4 P neighbours with weight 1/4, the
 * same pattern the shear term of grad_muu uses.
 ***********************************************************************/
static void ei_accum_modesep_at_sigma(modPar *mod, bndPar *bnd,
                                      float *fwd_txx, float *fwd_tzz,
                                      float *fwd_txz,
                                      wflPar *wfl_adj, float dt)
{
	int ix, iz, n1 = mod->naz, io = mod->iorder;
	int ibx, iex, ibz, iez;
	size_t i;
	float c1, c2, c3, c4;
	float sdx = 1.0f/mod->dx, sdz = 1.0f/mod->dz;
	float rdt = 1.0f/dt;

	ei_fdcoef(io, &c1, &c2, &c3, &c4);

	/* sigma_a = C psi_sigma.  l2m/lam at the P node, muu at the Txz node,
	 * matching where psi_txx/psi_tzz and psi_txz are defined. */
	for (i = 0; i < ei_sizem; i++) {
		float pxx = wfl_adj->txx[i], pzz = wfl_adj->tzz[i];
		ei_saxx[i] = mod->l2m[i]*pxx + mod->lam[i]*pzz;
		ei_sazz[i] = mod->lam[i]*pxx + mod->l2m[i]*pzz;
		ei_saxz[i] = mod->muu[i]*wfl_adj->txz[i];
	}

	/* Accelerations on their native velocity nodes. */
	for (ix = mod->ioXx; ix < mod->ieXx; ix++) {
		for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
			int ig = ix*n1+iz;
			ei_afx[ig] = rdt * mod->rox[ig] *
			    ei_divtau_x(fwd_txx, fwd_txz, ig, n1, io, c1, c2, c3, c4);
			ei_aax[ig] = rdt * mod->rox[ig] *
			    ei_divtau_x(ei_saxx, ei_saxz, ig, n1, io, c1, c2, c3, c4);
		}
	}
	for (ix = mod->ioZx; ix < mod->ieZx; ix++) {
		for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
			int ig = ix*n1+iz;
			ei_afz[ig] = rdt * mod->roz[ig] *
			    ei_divtau_z(fwd_tzz, fwd_txz, ig, n1, io, c1, c2, c3, c4);
			ei_aaz[ig] = rdt * mod->roz[ig] *
			    ei_divtau_z(ei_sazz, ei_saxz, ig, n1, io, c1, c2, c3, c4);
		}
	}

	ei_ms_bounds(mod, bnd, 0, &ibx, &iex, &ibz, &iez);

	for (ix = ibx; ix < iex; ix++) {
		for (iz = ibz; iz < iez; iz++) {
			int ig = ix*n1+iz;
			float Ddf, Dda, Cdf, Cda, shear;

			/* P channel: Ddot = div a, native at the P node. */
			Ddf = ei_div_v(ei_afx, ei_afz, ix, iz, n1, sdx, sdz, io,
			               c1, c2, c3, c4);
			Dda = ei_div_v(ei_aax, ei_aaz, ix, iz, n1, sdx, sdz, io,
			               c1, c2, c3, c4);
			ei_K[ig] += dt * Ddf * Dda;

			/* S channel: Cdot = curl a, native at the Txz node. */
			Cdf = ei_curl_v(ei_afx, ei_afz, ix, iz, n1, sdx, sdz, io,
			                c1, c2, c3, c4);
			Cda = ei_curl_v(ei_aax, ei_aaz, ix, iz, n1, sdx, sdz, io,
			                c1, c2, c3, c4);
			shear = 0.25f * dt * Cdf * Cda;
			ei_KS[(ix-1)*n1+iz-1] += shear;
			ei_KS[ix*n1+iz-1]     += shear;
			ei_KS[(ix-1)*n1+iz]   += shear;
			ei_KS[ig]             += shear;
		}
	}
}


/***********************************************************************
 * ei_accum_modesep_at_vel -- the two POTENTIAL halves, V_P and V_S.
 *
 *     V_P = Vp^2 grad D_f . grad D_a,      V_S = Vs^2 grad C_f . grad C_a
 *
 * Two passes.  The first fills D = div v (P node) and C = curl v (Txz node)
 * over the WIDE box for both fields; the second differentiates them over the
 * NARROW box.  D and C have to be materialised because it is their gradient
 * that is needed -- this is the one array cost EI_MODESEP carries over
 * gamma=0.
 *
 * The adjoint STATE velocity is b*psi_v and rox/roz are dt*b/dx, so b is
 * recovered as rox*dx/dt -- identical to ei_accum_potential_gamma0, and
 * deliberately so (section 2e-ii).
 *
 * MATERIAL REGISTRATION.  grad D lands on the velocity nodes, so Vp^2 is
 * needed there: Vp^2 = (l+2m)*b, with l2m averaged from the two P neighbours
 * and b = rox*dx/dt.  Vs^2 = m*b likewise, but muu lives on the Txz node, so
 * its two neighbours of a Vx node are offset in z and of a Vz node in x --
 * hence the different index pairs below.  Getting this wrong is a smooth
 * multiplicative error on one half only, which would bias the measured angle
 * rather than show up as noise, so it is worth the care.
 ***********************************************************************/
static void ei_accum_modesep_at_vel(modPar *mod, bndPar *bnd,
                                    float *fwd_vx, float *fwd_vz,
                                    wflPar *wfl_adj, float dt)
{
	int ix, iz, n1 = mod->naz, io = mod->iorder;
	int ibx, iex, ibz, iez;
	int wbx, wex, wbz, wez;
	size_t i;
	float c1, c2, c3, c4;
	float sdx = 1.0f/mod->dx, sdz = 1.0f/mod->dz;
	float bscx = mod->dx/dt, bscz = mod->dz/dt;

	ei_fdcoef(io, &c1, &c2, &c3, &c4);

	for (i = 0; i < ei_sizem; i++) {
		ei_vax[i] = wfl_adj->vx[i] * mod->rox[i] * bscx;
		ei_vaz[i] = wfl_adj->vz[i] * mod->roz[i] * bscz;
	}

	/* Pass 1: the two Helmholtz scalars, on the wide box. */
	ei_ms_bounds(mod, bnd, 1, &wbx, &wex, &wbz, &wez);
	for (ix = wbx; ix < wex; ix++) {
		for (iz = wbz; iz < wez; iz++) {
			int ig = ix*n1+iz;
			ei_Df[ig] = ei_div_v (fwd_vx, fwd_vz, ix, iz, n1, sdx, sdz, io,
			                      c1, c2, c3, c4);
			ei_Da[ig] = ei_div_v (ei_vax, ei_vaz, ix, iz, n1, sdx, sdz, io,
			                      c1, c2, c3, c4);
			ei_Cf[ig] = ei_curl_v(fwd_vx, fwd_vz, ix, iz, n1, sdx, sdz, io,
			                      c1, c2, c3, c4);
			ei_Ca[ig] = ei_curl_v(ei_vax, ei_vaz, ix, iz, n1, sdx, sdz, io,
			                      c1, c2, c3, c4);
		}
	}

	/* Pass 2: pair the gradients and scatter onto the P grid. */
	ei_ms_bounds(mod, bnd, 0, &ibx, &iex, &ibz, &iez);
	for (ix = ibx; ix < iex; ix++) {
		for (iz = ibz; iz < iez; iz++) {
			int ig = ix*n1+iz;
			float bx, bz, vp2, vs2, gf, ga, prod;

			/* --- Vx node (ix-1/2, iz): dx(D) and dz(C) --- */
			bx  = mod->rox[ig] * bscx;
			vp2 = 0.5f*(mod->l2m[ig-n1] + mod->l2m[ig]) * bx;
			vs2 = 0.5f*(mod->muu[ig]    + mod->muu[ig+1]) * bx;

			gf = sdx * ei_dm(ei_Df, ig, n1, io, c1, c2, c3, c4);
			ga = sdx * ei_dm(ei_Da, ig, n1, io, c1, c2, c3, c4);
			prod = 0.5f * dt * vp2 * gf * ga;
			ei_P[ig-n1] += prod;
			ei_P[ig]    += prod;

			gf = sdz * ei_dp(ei_Cf, ig, 1, io, c1, c2, c3, c4);
			ga = sdz * ei_dp(ei_Ca, ig, 1, io, c1, c2, c3, c4);
			prod = 0.5f * dt * vs2 * gf * ga;
			ei_VS[ig-n1] += prod;
			ei_VS[ig]    += prod;

			/* --- Vz node (ix, iz-1/2): dz(D) and dx(C) --- */
			bz  = mod->roz[ig] * bscz;
			vp2 = 0.5f*(mod->l2m[ig-1] + mod->l2m[ig])    * bz;
			vs2 = 0.5f*(mod->muu[ig]   + mod->muu[ig+n1]) * bz;

			gf = sdz * ei_dm(ei_Df, ig, 1, io, c1, c2, c3, c4);
			ga = sdz * ei_dm(ei_Da, ig, 1, io, c1, c2, c3, c4);
			prod = 0.5f * dt * vp2 * gf * ga;
			ei_P[ig-1] += prod;
			ei_P[ig]   += prod;

			gf = sdx * ei_dp(ei_Cf, ig, n1, io, c1, c2, c3, c4);
			ga = sdx * ei_dp(ei_Ca, ig, n1, io, c1, c2, c3, c4);
			prod = 0.5f * dt * vs2 * gf * ga;
			ei_VS[ig-1] += prod;
			ei_VS[ig]   += prod;
		}
	}
}


/***********************************************************************
 * energyImagingAccumAtSigma -- the half that needs psi_SIGMA, one time step.
 *
 * MUST be called where psi_sigma is the valid multiplier, i.e. BEFORE
 * callAdjKernel -- the same point in the backward loop as accumGradient.
 *
 * gamma=2: accumulates the POTENTIAL half, sigma_fwd : psi_sigma (below).
 * gamma=0: accumulates the KINETIC half instead, because at gamma=0 it is K
 *          that is built from stresses and V_0 that is built from velocities.
 *          The two halves swap call sites; that is why these functions are
 *          named for WHEN they run, not for what they compute.
 *
 * txx/tzz are native at the P grid.  txz is native at the Txz grid and is
 * scattered to the 4 surrounding P points with weight 0.25, matching the
 * shear part of grad_muu in fwi_gradient.c.  There is NO factor 2 on the
 * shear term: the 2 of sig_ij eps_ij is carried by the 1/mu Voigt weight,
 * which Q^-1 has already inverted away.
 *
 * fdelmodc stores T_ij = -sigma_ij; the sign cancels in a stress-stress
 * product, so no correction is needed.
 ***********************************************************************/
void energyImagingAccumAtSigma(modPar *mod, bndPar *bnd,
                               float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                               wflPar *wfl_adj, float dt)
{
	int ix, iz, n1;
	int ibPx, iePx, ibPz, iePz, ibTx, ieTx, ibTz, ieTz;

	/* Both accumulators are checked, not just the one this call site feeds:
	 * which of ei_K / ei_P that is depends on ei_gamma. */
	if (ei_mode == EI_OFF || !ei_K || !ei_P) return;
	if (!fwd_txx || !fwd_tzz || !fwd_txz) return;

	n1 = mod->naz;
	ei_bounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz, &ibTx, &ieTx, &ibTz, &ieTz);

	if (ei_gamma == EI_MODESEP) {
		ei_accum_modesep_at_sigma(mod, bnd, fwd_txx, fwd_tzz, fwd_txz,
		                          wfl_adj, dt);
		return;
	}

	if (ei_gamma == EI_GAMMA0) {
		ei_accum_kinetic_gamma0(mod, fwd_txx, fwd_tzz, fwd_txz, wfl_adj, dt);
		return;
	}

	/* Normal stresses: both native at P grid. */
	for (ix = ibPx; ix < iePx; ix++) {
		for (iz = ibPz; iz < iePz; iz++) {
			int ig = ix*n1+iz;
			ei_P[ig] += dt * ( fwd_txx[ig] * wfl_adj->txx[ig]
			                 + fwd_tzz[ig] * wfl_adj->tzz[ig] );
		}
	}

	/* Shear stress: native at Txz grid, scattered to the 4 P neighbours. */
	for (ix = ibTx; ix < ieTx; ix++) {
		for (iz = ibTz; iz < ieTz; iz++) {
			int ig = ix*n1+iz;
			float shear = 0.25f * dt * fwd_txz[ig] * wfl_adj->txz[ig];

			ei_P[(ix-1)*n1+iz-1] += shear;
			ei_P[ix*n1+iz-1]     += shear;
			ei_P[(ix-1)*n1+iz]   += shear;
			ei_P[ig]             += shear;
		}
	}
}


/***********************************************************************
 * energyImagingAccumAtVel -- the half that needs psi_V, one time step.
 *
 * MUST be called where psi_v is the valid multiplier, i.e. AFTER
 * callAdjKernel -- the same point in the backward loop as
 * accumGradient_rho_Dsig.
 *
 * gamma = 0 delegates to ei_accum_potential_gamma0 (V_0 is built from
 * VELOCITY gradients, so at gamma = 0 this call site carries the potential
 * half, not the kinetic one).  The gamma = 2 body below is K = v_f . psi_v.
 *
 * vx is native at the Vx grid, which straddles P(ix-1,iz) and P(ix,iz);
 * vz is native at the Vz grid, straddling P(ix,iz-1) and P(ix,iz).  Both are
 * scattered with weight 0.5, the same pattern as grad_rho in fwi_gradient.c,
 * but WITHOUT the rho factor: Q^-1 has cancelled it.
 *
 * Potential and kinetic are therefore sampled half a time step apart.  At
 * 11 Hz with dt = 1 ms that is a phase error of w*dt/2 = 0.035 rad, which is
 * immaterial for a weight bounded in [0,1].
 ***********************************************************************/
void energyImagingAccumAtVel(modPar *mod, bndPar *bnd,
                             float *fwd_vx, float *fwd_vz,
                             wflPar *wfl_adj, float dt)
{
	int ix, iz, n1;

	if (ei_mode == EI_OFF || !ei_K || !ei_P) return;
	if (!fwd_vx || !fwd_vz) return;

	n1 = mod->naz;

	if (ei_gamma == EI_MODESEP) {
		ei_accum_modesep_at_vel(mod, bnd, fwd_vx, fwd_vz, wfl_adj, dt);
		return;
	}

	if (ei_gamma == EI_GAMMA0) {
		ei_accum_potential_gamma0(mod, bnd, fwd_vx, fwd_vz, wfl_adj, dt);
		return;
	}

	(void)bnd;

	for (ix = mod->ioXx; ix < mod->ieXx; ix++) {
		for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
			int ig = ix*n1+iz;
			float prod = 0.5f * dt * fwd_vx[ig] * wfl_adj->vx[ig];
			ei_K[(ix-1)*n1+iz] += prod;
			ei_K[ig]           += prod;
		}
	}
	for (ix = mod->ioZx; ix < mod->ieZx; ix++) {
		for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
			int ig = ix*n1+iz;
			float prod = 0.5f * dt * fwd_vz[ig] * wfl_adj->vz[ig];
			ei_K[ig-1] += prod;
			ei_K[ig]   += prod;
		}
	}
}


/***********************************************************************
 * ei_build_weight -- Build a_tomo or a_migr on the P grid.
 *
 * The single place where the two combinations are formed, so the field
 * energyImagingGet writes to disk is by construction the field
 * energyImagingApply multiplies into the gradient.
 *
 * K and P are used AS THEY ARE ACCUMULATED:
 *
 *     a_tomo = |K+P| / (|K+P| + |K-P| + eps)
 *     a_migr = |K-P| / (|K+P| + |K-P| + eps)          (= cos^2(theta) = c2)
 *
 * They are NOT rescaled to equal global RMS first.  That rescaling used to be
 * applied here, and it was wrong: it multiplies the two halves by DIFFERENT
 * constants sK != sP, so at a point where the halves are physically equal
 * (K = P, equipartition -- i.e. perfect transmission, the one place the mask
 * must return 1) it makes them unequal and manufactures a floor
 *
 *     c2_floor = |r-1| / (|r+1| + |r-1|),      r = sK/sP = rms(P)/rms(K)
 *
 * out of nothing.  On the long-offset transmission test r = 0.766, giving
 * c2_floor = 0.119 against a measured corridor floor of 0.135 -- i.e. nearly
 * all of it.  Through exp(-alpha c2) that floor is a blanket ~2x dimming of
 * the whole transmission kernel.
 *
 * The ratio is still COMPUTED AND REPORTED, because it is the cheap check that
 * the two accumulators are commensurate at all (a relative mis-scaling has
 * bitten this file twice: the dx^2 mismatch between the gamma=0 halves, and
 * inserting rho and the compliance, which put it at 1e-13).  It is only no
 * longer APPLIED.  Reporting rather than dividing out is the point:
 * normalising made such a bug invisible in the log while it went on biasing
 * c2.  The absolute-amplitude check is test_ei_planewave.c, which validates
 * BOTH halves against rho A^2 w^2 dt sin^2(phi) pointwise.
 *
 * which = EI_GET_TOMO or EI_GET_MIGR.  Points outside the interior are set to
 * zero -- accumGradient writes nothing there.
 *
 * energy_alpha > 0 replaces the linear weight by Rocha's exponential mask
 * (eq 45),
 *
 *     a_tomo = exp(-alpha * c2)        a_migr = exp(-alpha * (1 - c2))
 *
 * which is NOT a partition of unity: a_tomo + a_migr != 1, and
 * g*a_tomo + g*a_migr != g.  The regression check in test_energy_ic.sh only
 * holds at alpha = 0.  The reason to pay that price is the converted-mode
 * residue: at gamma = 0 a P-S pair has c2 = 1/2 exactly (V_0 vanishes
 * identically for a converted pair, so |k+p| = |k-p| = |k|), which the linear
 * weight HALF-PASSES into both outputs.  Only a mask that is small at
 * c2 = 1/2 rejects it, e.g. exp(-100/2) ~ 2e-22.  gamma = 0 puts the
 * conversions at a single known value of c2; alpha is what then removes them.
 *
 * Returns the number of imaged points, or 0 if either half is identically
 * zero.  *ratio receives rms(P)/rms(K) for reporting.
 ***********************************************************************/
static long ei_build_weight(modPar *mod, bndPar *bnd, float *w, int which,
                            double *ratio,
                            int *ibPx_o, int *iePx_o, int *ibPz_o, int *iePz_o)
{
	int ix, iz, n1 = mod->naz;
	int ibPx, iePx, ibPz, iePz, ibTx, ieTx, ibTz, ieTz;
	double sum_k2 = 0.0, sum_p2 = 0.0, sum_d = 0.0;
	long npt = 0;
	float sK, sP, eps;

	ei_bounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz, &ibTx, &ieTx, &ibTz, &ieTz);
	*ibPx_o = ibPx; *iePx_o = iePx; *ibPz_o = ibPz; *iePz_o = iePz;

	for (ix = ibPx; ix < iePx; ix++) {
		for (iz = ibPz; iz < iePz; iz++) {
			int ig = ix*n1+iz;
			sum_k2 += (double)ei_K[ig] * ei_K[ig];
			sum_p2 += (double)ei_P[ig] * ei_P[ig];
			npt++;
		}
	}
	if (npt == 0 || sum_k2 <= 0.0 || sum_p2 <= 0.0) return 0;
	sK = (float)(1.0 / sqrt(sum_k2 / (double)npt));
	sP = (float)(1.0 / sqrt(sum_p2 / (double)npt));
	*ratio = sK / sP;

	/* Mean denominator, for the eps floor. */
	for (ix = ibPx; ix < iePx; ix++) {
		for (iz = ibPz; iz < iePz; iz++) {
			int ig = ix*n1+iz;
			sum_d += fabsf(ei_K[ig] + ei_P[ig]) + fabsf(ei_K[ig] - ei_P[ig]);
		}
	}
	eps = EI_EPS_FRAC * (float)(sum_d / (double)npt);

	memset(w, 0, ei_sizem * sizeof(float));
	for (ix = ibPx; ix < iePx; ix++) {
		for (iz = ibPz; iz < iePz; iz++) {
			int ig = ix*n1+iz;
			float tomo = fabsf(ei_K[ig] + ei_P[ig]);
			float migr = fabsf(ei_K[ig] - ei_P[ig]);

			if (ei_alpha > 0.0f) {
				/* c2 = cos^2(theta).  Note the eps sits in the
				 * denominator only, so a dead point (tomo = migr = 0)
				 * gets c2 = 0 and hence a_tomo = 1, a_migr = exp(-alpha).
				 * Harmless -- the gradient is zero there anyway -- but it
				 * is why the linear branch below is kept in its exact
				 * original form rather than written as 1 - c2. */
				float c2 = migr / (tomo + migr + eps);
				w[ig] = expf(-ei_alpha * ((which == EI_GET_MIGR)
				                          ? (1.0f - c2) : c2));
			}
			else {
				w[ig] = ((which == EI_GET_MIGR) ? migr : tomo)
				        / (tomo + migr + eps);
			}
		}
	}
	return npt;
}


/***********************************************************************
 * ei_build_weight_modesep -- Build one of the three EI_MODESEP weights.
 *
 *     t_P = |K_P+V_P|   m_P = |K_P-V_P|   E_P = t_P + m_P     (and S)
 *
 *     a_tomo^PP  = t_P / (E_P + eps)
 *     a_tomo^SS  = t_S / (E_S + eps)
 *     a_tomo^mix = (t_P + t_S) / (E_P + E_S + eps)
 *
 * The mixed weight is NOT a new choice: it is identically the energy-weighted
 * average of the two per-mode weights,
 *
 *     (E_P a^PP + E_S a^SS) / (E_P + E_S)  =  (t_P + t_S) / (E_P + E_S),
 *
 * so the only decision made here is to weight each mode by how much of it is
 * present, which is the one weighting that needs no justification.  Each of
 * the three is separately a partition of unity with its migration twin, so
 * the g*a_tomo + g*a_migr = g regression check holds per parameter even
 * though g_lam and g_mu receive DIFFERENT weights (section 2e, and
 * energyImagingApply below).
 *
 * As in ei_build_weight the halves are combined exactly as accumulated, with
 * no rescaling to equal RMS -- see the long note there for why that rescaling
 * is wrong.  The two ratios rms(V_P)/rms(K_P) and rms(V_S)/rms(K_S) are
 * reported instead; both must be O(1), and for EI_MODESEP they have a sharper
 * meaning than at gamma=0 because K and V are provably equal POINTWISE for a
 * single plane wave of either mode (the Vp and Vs cancel, section 2e).
 *
 * Returns the number of imaged points, or 0 if a needed half is empty.
 ***********************************************************************/
static long ei_build_weight_modesep(modPar *mod, bndPar *bnd, float *w,
                                    int which, double *ratP, double *ratS,
                                    int *ibx_o, int *iex_o,
                                    int *ibz_o, int *iez_o)
{
	int ix, iz, n1 = mod->naz;
	int ibx, iex, ibz, iez;
	int useP = (which != EI_GET_TOMO_S && which != EI_GET_MIGR_S);
	int useS = (which != EI_GET_TOMO_P && which != EI_GET_MIGR_P);
	int ismigr = (which == EI_GET_MIGR || which == EI_GET_MIGR_P ||
	              which == EI_GET_MIGR_S);
	double sKP = 0.0, sVP = 0.0, sKS = 0.0, sVS = 0.0, sum_d = 0.0;
	long npt = 0;
	float eps;

	ei_ms_bounds(mod, bnd, 0, &ibx, &iex, &ibz, &iez);
	*ibx_o = ibx; *iex_o = iex; *ibz_o = ibz; *iez_o = iez;

	for (ix = ibx; ix < iex; ix++) {
		for (iz = ibz; iz < iez; iz++) {
			int ig = ix*n1+iz;
			sKP += (double)ei_K[ig]  * ei_K[ig];
			sVP += (double)ei_P[ig]  * ei_P[ig];
			sKS += (double)ei_KS[ig] * ei_KS[ig];
			sVS += (double)ei_VS[ig] * ei_VS[ig];
			sum_d += (useP ? fabsf(ei_K[ig] + ei_P[ig])
			               + fabsf(ei_K[ig] - ei_P[ig]) : 0.0f)
			       + (useS ? fabsf(ei_KS[ig] + ei_VS[ig])
			               + fabsf(ei_KS[ig] - ei_VS[ig]) : 0.0f);
			npt++;
		}
	}
	if (npt == 0 || sum_d <= 0.0) return 0;

	*ratP = (sKP > 0.0) ? sqrt(sVP / sKP) : 0.0;
	*ratS = (sKS > 0.0) ? sqrt(sVS / sKS) : 0.0;

	eps = EI_EPS_FRAC * (float)(sum_d / (double)npt);

	memset(w, 0, ei_sizem * sizeof(float));
	for (ix = ibx; ix < iex; ix++) {
		for (iz = ibz; iz < iez; iz++) {
			int ig = ix*n1+iz;
			float tomo = 0.0f, migr = 0.0f;

			if (useP) {
				tomo += fabsf(ei_K[ig] + ei_P[ig]);
				migr += fabsf(ei_K[ig] - ei_P[ig]);
			}
			if (useS) {
				tomo += fabsf(ei_KS[ig] + ei_VS[ig]);
				migr += fabsf(ei_KS[ig] - ei_VS[ig]);
			}

			if (ei_alpha > 0.0f) {
				float c2 = migr / (tomo + migr + eps);
				w[ig] = expf(-ei_alpha * (ismigr ? (1.0f - c2) : c2));
			}
			else {
				w[ig] = (ismigr ? migr : tomo) / (tomo + migr + eps);
			}
		}
	}
	return npt;
}


/***********************************************************************
 * energyImagingApply -- Build the angle weight and apply it in place.
 *
 * Call once per shot, AFTER adj_shot, while the gradient is still in Lame
 * space.  The weight is a single scalar per grid point and the Lame ->
 * velocity chain rule is pointwise linear in (g_lam, g_mu, g_rho), so
 * weighting before or after convertGradientToVelocity gives the same answer.
 *
 * The same weight is applied to all three parameter gradients.  The lambda,
 * mu and rho kernels do not share an identical angle dependence, so this is
 * one filter applied to three kernels, not three matched filters.
 *
 * energy_ic=EI_QC builds and reports the weight but leaves the gradient
 * untouched.
 ***********************************************************************/
void energyImagingApply(modPar *mod, bndPar *bnd,
                        float *grad_lam, float *grad_muu, float *grad_rho,
                        int verbose)
{
	int ix, iz, n1;
	int ibPx, iePx, ibPz, iePz;
	double w_sum = 0.0, ratio;
	long   npt;
	int    wtomo = (ei_mode == EI_MIGR) ? EI_GET_MIGR : EI_GET_TOMO;

	if (ei_mode == EI_OFF || !ei_K || !ei_P) return;

	n1 = mod->naz;

	/* EI_MODESEP applies TWO different weights: g_lam is a pure-PP kernel
	 * (sigma_kk is blind to S), so it gets the P-channel weight, while g_mu
	 * and g_rho carry all four mode pairs and get the energy-weighted mix.
	 * Each weight is still a partition of unity with its own migration twin,
	 * so the g*a_tomo + g*a_migr = g check holds per parameter. */
	if (ei_gamma == EI_MODESEP) {
		double ratP, ratS;
		int    wP = (ei_mode == EI_MIGR) ? EI_GET_MIGR_P : EI_GET_TOMO_P;

		npt = ei_build_weight_modesep(mod, bnd, ei_w, wtomo, &ratP, &ratS,
		                              &ibPx, &iePx, &ibPz, &iePz);
		if (npt == 0) {
			vwarn("energyImagingApply: EI_MODESEP accumulated nothing. "
			      "Leaving the gradient unweighted.");
			return;
		}

		if (verbose) {
			for (ix = ibPx; ix < iePx; ix++)
				for (iz = ibPz; iz < iePz; iz++) w_sum += ei_w[ix*n1+iz];
			vmess("  energy_ic: rms(V_P)/rms(K_P) = %.4g, "
			      "rms(V_S)/rms(K_S) = %.4g  (both should be O(1); K and V "
			      "are equal POINTWISE for a single plane wave of either "
			      "mode, so a value far from 1 is a real scaling problem)",
			      ratP, ratS);
			vmess("  energy_ic: mean %s weight (mixed, applied to g_mu/g_rho) "
			      "= %.4f over %ld imaged points",
			      (ei_mode == EI_MIGR) ? "migration" : "tomographic",
			      w_sum / (double)npt, npt);
		}

		if (ei_mode == EI_QC) return;

		/* g_mu and g_rho take the mixed weight currently in ei_w. */
		for (ix = ibPx; ix < iePx; ix++) {
			for (iz = ibPz; iz < iePz; iz++) {
				int ig = ix*n1+iz;
				if (grad_muu) grad_muu[ig] *= ei_w[ig];
				if (grad_rho) grad_rho[ig] *= ei_w[ig];
			}
		}

		/* g_lam takes the P-channel weight. */
		if (grad_lam) {
			ei_build_weight_modesep(mod, bnd, ei_w, wP, &ratP, &ratS,
			                        &ibPx, &iePx, &ibPz, &iePz);
			for (ix = ibPx; ix < iePx; ix++)
				for (iz = ibPz; iz < iePz; iz++)
					grad_lam[ix*n1+iz] *= ei_w[ix*n1+iz];
		}

		/* Points outside the (narrower) EI_MODESEP box were never weighted
		 * above; zero them so the imaged region is the region the weight was
		 * actually defined on, exactly as the other gammas do by construction. */
		{
			int wbx, wex, wbz, wez;
			size_t i;
			ei_ms_bounds(mod, bnd, 0, &wbx, &wex, &wbz, &wez);
			for (i = 0; i < ei_sizem; i++) {
				int cx = (int)(i / (size_t)n1), cz = (int)(i % (size_t)n1);
				if (cx >= wbx && cx < wex && cz >= wbz && cz < wez) continue;
				if (grad_lam) grad_lam[i] = 0.0f;
				if (grad_muu) grad_muu[i] = 0.0f;
				if (grad_rho) grad_rho[i] = 0.0f;
			}
		}
		return;
	}

	npt = ei_build_weight(mod, bnd, ei_w, wtomo,
	                      &ratio, &ibPx, &iePx, &ibPz, &iePz);
	if (npt == 0) {
		vwarn("energyImagingApply: one half of the energy inner product is "
		      "identically zero. Leaving the gradient unweighted.");
		return;
	}

	if (verbose) {
		for (ix = ibPx; ix < iePx; ix++)
			for (iz = ibPz; iz < iePz; iz++) w_sum += ei_w[ix*n1+iz];
		vmess("  energy_ic: rms(P)/rms(K) = %.4g  (1 = kinetic/potential "
		      "equipartition; DIAGNOSTIC ONLY -- the halves are combined "
		      "unscaled, so a value far from 1 is a real commensurability "
		      "problem, not something that gets divided out)", ratio);
		vmess("  energy_ic: mean %s weight = %.4f over %ld imaged points",
		      (ei_mode == EI_MIGR) ? "migration" : "tomographic",
		      w_sum / (double)npt, npt);
	}

	if (ei_mode == EI_QC) return;

	for (ix = ibPx; ix < iePx; ix++) {
		for (iz = ibPz; iz < iePz; iz++) {
			int ig = ix*n1+iz;
			if (grad_lam) grad_lam[ig] *= ei_w[ig];
			if (grad_muu) grad_muu[ig] *= ei_w[ig];
			if (grad_rho) grad_rho[ig] *= ei_w[ig];
		}
	}
}


/***********************************************************************
 * energyImagingGet -- Copy one diagnostic field onto a caller array.
 *
 * which: one of the EI_GET_* codes (see fdelfwi.h).
 *
 * Every weight is rebuilt from the stored accumulators with exactly the same
 * normalisation and the same interior bounds as energyImagingApply, so the
 * field written to disk is the field that was applied.  That also means the
 * complementary weight is available even when only one of them was used, and
 * that both are available at energy_ic=EI_QC.
 *
 * At EI_MODESEP the four legacy codes keep their meaning by mapping onto the
 * P channel (EI_GET_KIN -> K_P, EI_GET_POT -> V_P) and onto the MIXED weight
 * (EI_GET_TOMO/MIGR), which is the weight actually applied to g_mu and g_rho.
 * The six new codes expose the S channel and the per-mode weights.  The S and
 * per-mode codes return 0 at the gamma-family settings, where no such
 * decomposition exists.
 *
 * Returns 1 on success, 0 if nothing has been accumulated or if the requested
 * field does not exist for the current energy_gamma.
 ***********************************************************************/
int energyImagingGet(modPar *mod, bndPar *bnd, int which, float *out)
{
	int ibPx, iePx, ibPz, iePz;
	double ratio;

	if (ei_mode == EI_OFF || !ei_K || !ei_P || !out) return 0;

	if (ei_gamma == EI_MODESEP) {
		double ratP, ratS;

		if (!ei_KS || !ei_VS) return 0;

		switch (which) {
			case EI_GET_KIN:   /* P-channel kinetic   */
				memcpy(out, ei_K,  ei_sizem * sizeof(float)); return 1;
			case EI_GET_POT:   /* P-channel potential */
				memcpy(out, ei_P,  ei_sizem * sizeof(float)); return 1;
			case EI_GET_KIN_S:
				memcpy(out, ei_KS, ei_sizem * sizeof(float)); return 1;
			case EI_GET_POT_S:
				memcpy(out, ei_VS, ei_sizem * sizeof(float)); return 1;
			default:
				break;
		}
		return ei_build_weight_modesep(mod, bnd, out, which, &ratP, &ratS,
		                               &ibPx, &iePx, &ibPz, &iePz) ? 1 : 0;
	}

	/* gamma family: no mode decomposition, so the S and per-mode codes do
	 * not exist.  Report nothing rather than something misleading. */
	if (which == EI_GET_KIN_S  || which == EI_GET_POT_S  ||
	    which == EI_GET_TOMO_P || which == EI_GET_MIGR_P ||
	    which == EI_GET_TOMO_S || which == EI_GET_MIGR_S) return 0;

	if (which == EI_GET_KIN) {
		memcpy(out, ei_K, ei_sizem * sizeof(float));
		return 1;
	}
	if (which == EI_GET_POT) {
		memcpy(out, ei_P, ei_sizem * sizeof(float));
		return 1;
	}

	return ei_build_weight(mod, bnd, out, which, &ratio,
	                       &ibPx, &iePx, &ibPz, &iePz) ? 1 : 0;
}


void energyImagingFree(void)
{
	if (ei_K)   free(ei_K);
	if (ei_P)   free(ei_P);
	if (ei_w)   free(ei_w);
	ei_K = ei_P = ei_w = NULL;

	if (ei_saxx) free(ei_saxx);
	if (ei_sazz) free(ei_sazz);
	if (ei_saxz) free(ei_saxz);
	if (ei_vax)  free(ei_vax);
	if (ei_vaz)  free(ei_vaz);
	ei_saxx = ei_sazz = ei_saxz = ei_vax = ei_vaz = NULL;

	/* EI_MODESEP scratch.  energyImagingInit calls this first, so leaving a
	 * stale non-NULL pointer here would be a dangling-pointer bug, not just
	 * a leak. */
	if (ei_KS)  free(ei_KS);
	if (ei_VS)  free(ei_VS);
	ei_KS = ei_VS = NULL;

	if (ei_afx) free(ei_afx);
	if (ei_afz) free(ei_afz);
	if (ei_aax) free(ei_aax);
	if (ei_aaz) free(ei_aaz);
	ei_afx = ei_afz = ei_aax = ei_aaz = NULL;

	if (ei_Df)  free(ei_Df);
	if (ei_Da)  free(ei_Da);
	if (ei_Cf)  free(ei_Cf);
	if (ei_Ca)  free(ei_Ca);
	ei_Df = ei_Da = ei_Cf = ei_Ca = NULL;

	ei_sizem = 0;
	ei_mode  = EI_OFF;
	ei_gamma = EI_GAMMA2;
	ei_alpha = 0.0f;
}
