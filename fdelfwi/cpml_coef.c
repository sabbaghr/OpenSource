#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"

/*********************************************************************
 *
 * cpml_coef.c -- CFS-CPML coefficient construction, shared by the
 *                forward (boundaries.c) and the adjoint
 *                (boundaries_cpml_adj.c).
 *
 * The three coefficients of the CFS-CPML memory-variable recursion
 *
 *     psi^{n+1} = b * psi^n + c * D(u)
 *     D_stretched = ik * D(u) + psi^{n+1}
 *
 * depend only on POSITION, never on the wavefield:
 *
 *     sigma = sigma_max * xi^m          conductivity profile
 *     kappa = 1 + (kmax-1) * xi^m       real stretch
 *     alpha = amax * (1 - xi)           CFS frequency shift
 *
 *     b  = exp(-(sigma/kappa + alpha) * dt)
 *     c  = sigma/(kappa*(sigma+kappa*alpha)) * (b - 1)
 *     ik = 1/kappa
 *
 * with xi the normalised depth into the PML (0 at the interface, 1 at
 * the outer edge).  Outside the PML the defaults b=0, c=0, ik=1 make
 * the recursion a no-op, so the stretched derivative degenerates to
 * the plain one and psi stays identically zero.
 *
 * WHY THIS FILE EXISTS.  The adjoint needs exactly the same b/c/ik the
 * forward used -- the discrete transpose is only a transpose if the
 * coefficients match bit for bit.  Re-deriving them in the adjoint is a
 * drift hazard, made worse by the fact that the forward builds TWO
 * DIFFERENT sets:
 *
 *   boundariesP (velocity update) samples xi at the FIELD positions,
 *               xi = (ioXx - ix)/npml
 *   boundariesV (stress update)   samples xi half a cell off,
 *               xi = (ioXx - ix - 0.5)/npml   for the cell-centred grid
 *               and a face-staggered set alongside it
 *
 * The two sets are NOT interchangeable, and both are needed.  They are
 * ported verbatim from fdelmodc/boundaries.c; keep them diffable.
 *
 * NOTE amax.  amax=0 kills the CFS frequency shift and degenerates
 * CPML to a classical PML (poor at grazing incidence).  It is wired
 * through bndPar from getParameters.c -- do not hardcode it here.
 *
 **********************************************************************/


/* Scratch used by both builders. */
static void cpml_profile(double xi_val, double m_cpml, double sigma_max,
                         double kmax, double amax, double dt,
                         float *b, float *c, float *ik)
{
    double sig, kap, alp, sak;

    if (xi_val < 0.0) xi_val = 0.0;
    if (xi_val > 1.0) xi_val = 1.0;

    sig = sigma_max * pow(xi_val, m_cpml);
    kap = 1.0 + (kmax-1.0) * pow(xi_val, m_cpml);
    alp = amax * (1.0 - xi_val);
    sak = sig + kap*alp;

    *b  = (float)exp(-(sig/kap + alp)*dt);
    *c  = (sak > 1e-30) ? (float)(sig/(kap*sak)*((double)(*b)-1.0)) : 0.0f;
    *ik = (float)(1.0/kap);
}


static double cpml_sigma_max(modPar mod, bndPar bnd, double m_cpml)
{
    double R_cpml, L_pml;

    R_cpml = (bnd.R > 0.0) ? bnd.R : 1e-5;
    L_pml  = bnd.npml * mod.dx;

    return (m_cpml+1.0)*mod.cp_min/(2.0*L_pml)*log(1.0/R_cpml);
}


/*---------------------------------------------------------------------
 * cpmlCoefVel -- coefficients for the VELOCITY update (boundariesP).
 *
 * Face-staggered only.  x is sampled on the vx faces (mod.ioXx/ieXx),
 * z on the vz faces (mod.ioZz/ieZz).  Caller supplies arrays of length
 * mod.nax (x) and mod.naz (z); they are fully initialised here.
 *--------------------------------------------------------------------*/
void cpmlCoefVel(modPar mod, bndPar bnd,
                 float *b_xf, float *c_xf, float *ik_xf,
                 float *b_zf, float *c_zf, float *ik_zf)
{
    double m_cpml, sigma_max, kmax, amax;
    int    ix, iz, ib, npml, n1, n2;

    n1   = mod.naz;
    n2   = mod.nax;
    npml = bnd.npml;

    m_cpml    = (bnd.m > 0.0) ? bnd.m : 2.0;
    kmax      = bnd.kmax;
    amax      = bnd.amax;
    sigma_max = cpml_sigma_max(mod, bnd, m_cpml);

    /* default outside PML: b=0 c=0 ik=1 => no correction applied */
    for (ib=0; ib<n2; ib++) { b_xf[ib]=0.0f; c_xf[ib]=0.0f; ik_xf[ib]=1.0f; }
    for (ib=0; ib<n1; ib++) { b_zf[ib]=0.0f; c_zf[ib]=0.0f; ik_zf[ib]=1.0f; }

    /* Left PML: vx face positions */
    if (bnd.lef == 2) {
        for (ix=((mod.ioXx-npml)>0?(mod.ioXx-npml):0); ix<(mod.ioXx<n2?mod.ioXx:n2); ix++) {
            cpml_profile((double)(mod.ioXx - ix) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_xf[ix], &c_xf[ix], &ik_xf[ix]);
        }
    }
    /* Right PML: vx face positions */
    if (bnd.rig == 2) {
        for (ix=(mod.ieXx>0?mod.ieXx:0); ix<((mod.ieXx+npml)<n2?(mod.ieXx+npml):n2); ix++) {
            cpml_profile((double)(ix - mod.ieXx + 1) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_xf[ix], &c_xf[ix], &ik_xf[ix]);
        }
    }
    /* Top PML: vz face positions */
    if (bnd.top == 2) {
        for (iz=((mod.ioZz-npml)>0?(mod.ioZz-npml):0); iz<(mod.ioZz<n1?mod.ioZz:n1); iz++) {
            cpml_profile((double)(mod.ioZz - iz) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_zf[iz], &c_zf[iz], &ik_zf[iz]);
        }
    }
    /* Bottom PML: vz face positions */
    if (bnd.bot == 2) {
        for (iz=(mod.ieZz>0?mod.ieZz:0); iz<((mod.ieZz+npml)<n1?(mod.ieZz+npml):n1); iz++) {
            cpml_profile((double)(iz - mod.ieZz + 1) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_zf[iz], &c_zf[iz], &ik_zf[iz]);
        }
    }
}


/*---------------------------------------------------------------------
 * cpmlCoefStr -- coefficients for the STRESS update (boundariesV).
 *
 * Two grids: cell-centred (*_xc/*_zc) for txx/tzz, face-staggered
 * (*_xf/*_zf) for txz.  NOTE these face arrays are NOT the same as the
 * ones cpmlCoefVel builds -- the sampling is offset by half a cell.
 *
 * The *_i reference indices translate fdelmodc's index convention (it
 * shifts ioPx/ioPz/ioTx/ioTz by npml in getParameters.c) into
 * fdelfwi's (it does not; ~60 FWI call sites rely on the unshifted
 * meaning).  Without the translation the left loop starts at a
 * negative index -- an out-of-bounds write -- and the right loop is
 * offset by 2*npml so most of the right PML keeps b=0,c=0,ik=1, i.e.
 * no damping at all.
 *--------------------------------------------------------------------*/
void cpmlCoefStr(modPar mod, bndPar bnd,
                 float *b_xc, float *c_xc, float *ik_xc,
                 float *b_zc, float *c_zc, float *ik_zc,
                 float *b_xf, float *c_xf, float *ik_xf,
                 float *b_zf, float *c_zf, float *ik_zf)
{
    double m_cpml, sigma_max, kmax, amax;
    int    ix, iz, ib, npml, n1, n2;
    int    ioXx_ref, ieXx_ref, ioZz_ref, ieZz_ref;
    int    ioPx_i, iePx_i, ioPz_i, iePz_i;

    n1   = mod.naz;
    n2   = mod.nax;
    npml = bnd.npml;

    m_cpml    = (bnd.m > 0.0) ? bnd.m : 2.0;
    kmax      = bnd.kmax;
    amax      = bnd.amax;
    sigma_max = cpml_sigma_max(mod, bnd, m_cpml);

    /* default outside PML: b=0 c=0 ik=1 => no correction */
    for (ib=0; ib<n2; ib++) { b_xc[ib]=0.0f; c_xc[ib]=0.0f; ik_xc[ib]=1.0f; }
    for (ib=0; ib<n1; ib++) { b_zc[ib]=0.0f; c_zc[ib]=0.0f; ik_zc[ib]=1.0f; }
    for (ib=0; ib<n2; ib++) { b_xf[ib]=0.0f; c_xf[ib]=0.0f; ik_xf[ib]=1.0f; }
    for (ib=0; ib<n1; ib++) { b_zf[ib]=0.0f; c_zf[ib]=0.0f; ik_zf[ib]=1.0f; }

    ioPx_i = mod.ioPx + (bnd.lef==2 ? npml : 0);
    iePx_i = mod.iePx - (bnd.rig==2 ? npml : 0);
    ioPz_i = mod.ioPz + (bnd.top==2 ? npml : 0);
    iePz_i = mod.iePz - (bnd.bot==2 ? npml : 0);

    ioXx_ref = mod.ioXx;
    ieXx_ref = iePx_i;
    ioZz_ref = ioPz_i+1;
    ieZz_ref = mod.ieXz;

    /* Left PML */
    if (bnd.lef == 2) {
        for (ix=ioPx_i-npml; ix<ioXx_ref; ix++) {                  /* cell centre */
            cpml_profile(((double)(ioXx_ref - ix) - 0.5) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_xc[ix], &c_xc[ix], &ik_xc[ix]);
        }
        for (ix=(mod.ioXx-npml>0?mod.ioXx-npml:0); ix<(mod.ioXx<n2?mod.ioXx:n2); ix++) { /* face */
            cpml_profile(((double)(mod.ioXx - ix) - 0.5) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_xf[ix], &c_xf[ix], &ik_xf[ix]);
        }
    }
    /* Right PML */
    if (bnd.rig == 2) {
        for (ix=ieXx_ref; ix<(ieXx_ref+npml<n2?ieXx_ref+npml:n2); ix++) {
            cpml_profile(((double)(ix - ieXx_ref) + 0.5) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_xc[ix], &c_xc[ix], &ik_xc[ix]);
        }
        for (ix=(mod.ieXx<n2?mod.ieXx:n2); ix<(mod.ieXx+npml<n2?mod.ieXx+npml:n2); ix++) {
            cpml_profile(((double)(ix - mod.ieXx) + 0.5) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_xf[ix], &c_xf[ix], &ik_xf[ix]);
        }
    }
    /* Top PML */
    if (bnd.top == 2) {
        for (iz=ioPz_i-npml; iz<ioZz_ref; iz++) {
            cpml_profile(((double)(ioZz_ref - iz) - 0.5) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_zc[iz], &c_zc[iz], &ik_zc[iz]);
        }
        for (iz=(mod.ioXz-npml>0?mod.ioXz-npml:0); iz<(mod.ioXz<n1?mod.ioXz:n1); iz++) {
            cpml_profile(((double)(mod.ioXz - iz) - 0.5) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_zf[iz], &c_zf[iz], &ik_zf[iz]);
        }
    }
    /* Bottom PML */
    if (bnd.bot == 2) {
        for (iz=ieZz_ref; iz<(ieZz_ref+npml<n1?ieZz_ref+npml:n1); iz++) {
            cpml_profile(((double)(iz - ieZz_ref) + 0.5) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_zc[iz], &c_zc[iz], &ik_zc[iz]);
        }
        for (iz=(mod.ieXz<n1?mod.ieXz:n1); iz<(mod.ieXz+npml<n1?mod.ieXz+npml:n1); iz++) {
            cpml_profile(((double)(iz - mod.ieXz) + 0.5) / (double)npml,
                         m_cpml, sigma_max, kmax, amax, mod.dt,
                         &b_zf[iz], &c_zf[iz], &ik_zf[iz]);
        }
    }
}
