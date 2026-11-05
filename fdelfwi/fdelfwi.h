#ifndef FDELFWI_H
#define FDELFWI_H

#include<stdlib.h>
#include<stdio.h>
#include<math.h>

// New data structures for FDELFWI
/* --- Wavefield Structure (Dynamic Data) --- */
typedef struct _wflType { 
    /* Velocity Fields */
    float *vz;
    float *vx;
    /* Pressure / Stress Fields */
    float *p;
    float *txx;
    float *tzz;
    float *txz;
    /* Potentials (optional) */
    float *pp;
    float *ss;
    /* Memory Variables for Q-Attenuation */
    float *r;
    float *q;
} wflPar;


typedef struct _compType { /* Receiver Type */
	int vz;
	int vx;
	int p;
	int txx;
	int tzz;
	int txz;
	int dxvx;
	int dzvz;
	int pp;
	int ss;
	int ud;
	int q;
} compType;

typedef struct _receiverPar { /* Receiver Parameters */
	char *file_rcv;
	compType type;
	int n;
	int nt;
	int delay;
	int skipdt;
	int max_nrec;
	int *z;
	int *x;
	float *zr;
	float *xr;
	int int_p;
	int int_vx;
	int int_vz;
	int aper_taper;	/* FWI: adjoint-injection taper width at the array ends,
					 * in grid points (0 = off). See aperture_taper.c */
	int scale;
	int sinkdepth;
	int sinkvel;
	float cp;
	float rho;
} recPar;

typedef struct _snapshotPar { /* Snapshot Parameters */
	char *file_snap;
	char *file_beam;
	compType type;
	int nsnap;
	int delay;
	float t0;
	int skipdt;
	int skipdz;
	int skipdx;
	int nz;
	int nx;
	int z1;
	int z2;
	int x1;
	int x2;
	int vxvztime;
	int beam;
	int withbnd;
} snaPar;

/* --- Model Parameters (Metadata + Data) --- */
typedef struct _modelPar { 
    int iorder;
    int ischeme; /* 1=Acoustic, 2=ViscoAc, 3=Elastic, 4=ViscoEl, 5=DoubleCpl */
    int grid_dir;
    int sh;
    
    /* Filenames */
    char *file_cp;
    char *file_ro;
    char *file_cs;
    char *file_qp;
    char *file_qs;
    
    /* Grid Dimensions */
    float dz; float dx; float dt;
    float tmod;
    int nt;
    float z0; float x0; float t0;
    
    /* Min/Max values */
    float cp_min; float cp_max;
    float cs_min; float cs_max;
    float ro_min; float ro_max;
    
    /* Grid Sizes */
    int nz; int nx;
    int naz; int nax;
    
    /* Offsets for Staggered Grid/PML */
    int ioXx; int ioXz; int ieXx; int ieXz; /* Vx */
    int ioZx; int ioZz; int ieZx; int ieZz; /* Vz */
    int ioPx; int ioPz; int iePx; int iePz; /* P/Txx/Tzz */
    int ioTx; int ioTz; int ieTx; int ieTz; /* Txz */
    
    /* Q-factor parameters */
    float Qp; float Qs;
    float fw; float qr;

    /* --- DATA ARRAYS --- */
    /* Raw Models (Optional storage) */
    float *cp; 
    float *cs; 
    float *rho;
    
    /* Finite Difference Coefficients */
    float *rox; /* Buoyancy X */
    float *roz; /* Buoyancy Z */
    float *l2m; /* Lambda + 2Mu (P-wave modulus) */
    float *lam; /* Lambda */
    float *muu; /* Shear Modulus (Mu) */
    
    /* Memory Variables for Q */
    float *tss; /* Tau-Sigma (P) */
    float *tep; /* Tau-Epsilon (P) */
    float *tes; /* Tau-Epsilon (S) */

} modPar;

typedef struct _waveletPar { /* Wavelet Parameters */
	char *file_src; /* general source */
	int nsrcf;
	int nt;
	int ns;
	int nx;
	float dt;
	float ds;
	float fmax;
	int random;
	int seed;
	int nst;
	size_t *nsamp;
	float *cp;
} wavPar;

typedef struct _sourcePar { /* Source Array Parameters */
	int n;
	int type;
	int orient;
	int *z;
	int *x;
	float Mxx;
	float Mzz;
	float Mxz;
	int single;	
	int plane;
	int circle;
	int array;
	int random;
	float *tbeg;
	float *tend;
	int multiwav;
	float angle;
	float velo;
	float amplitude;
	float dip;
	float strike;
	int distribution;
	int window;
    int injectionrate;
	int sinkdepth;
	int src_at_rcv; /* Indicates that wavefield should be injected at receivers */
    float wx;
    float wz;
} srcPar;

typedef struct _shotPar { /* Shot Parameters */
	int n;
	int *z;
	int *x;
} shotPar;

typedef struct _boundPar { /* Boundary Parameters */
	int top;
	int bot;
	int lef;
	int rig;
	float *tapz;
	float *tapx;
	float *tapxz;
	int cfree;
	int ntap;
	int *surface;
    int npml;
    float R; /* reflection at side of model */
    float m; /* scaling order */
    float *pml_Vx;
    float *pml_nzVx;
    float *pml_nxVz;
    float *pml_nzVz;
    float *pml_nxP;
    float *pml_nzP;

} bndPar;

/* --- Checkpoint Storage for FWI (disk-based, re-propagation approach) ---
 *
 * Saves the complete wavefield state (vx, vz, txx, tzz, txz) at a few
 * checkpoint times so the forward simulation can be restarted from any
 * checkpoint during the adjoint pass.  Between checkpoints the forward
 * wavefield is re-propagated and cross-correlated with the adjoint
 * wavefield at every time step to obtain the exact gradient.
 */
typedef struct _checkpointPar {
    int    nsnap;           /* Total number of stored checkpoints */
    int    skipdt;          /* Time steps between checkpoints */
    int    delay;           /* First checkpoint time step (usually 0) */
    int    naz;             /* Vertical array dimension (= mod->naz) */
    int    nax;             /* Horizontal array dimension (= mod->nax) */
    int    ischeme;         /* Wave equation type (1=acoustic, 3=elastic, etc.) */
    char   file_vx[1024];  /* Path for vx checkpoint file */
    char   file_vz[1024];  /* Path for vz checkpoint file */
    char   file_tzz[1024]; /* Path for tzz (pressure) checkpoint file */
    char   file_txx[1024]; /* Path for txx checkpoint file (elastic only) */
    char   file_txz[1024]; /* Path for txz checkpoint file (elastic only) */
    int    it;              /* Running snapshot index (for adjoint sync) */
} checkpointPar;

/* --- Misfit Function Type (extensible enum) --- */
typedef enum {
    MISFIT_L2 = 0,          /* L2 norm: J = 0.5*sum(r^2)*dt, adj = d_obs - d_syn */
    MISFIT_CORRELATION,      /* Cross-correlation (TODO) */
    MISFIT_KH                /* Kirchhoff-Helmholtz energy flux (kh_misfit.c) */
} misfitType;

/* --- Adjoint Source Parameters (for FWI backpropagation) --- */
typedef struct _adjSrcPar {
    int     nsrc;       /* Number of adjoint source traces (receivers) */
    int     nt;         /* Number of time samples per trace */
    size_t *xi;         /* Horizontal grid indices (staggered-grid aware) */
    size_t *zi;         /* Vertical grid indices (staggered-grid aware) */
    int    *typ;        /* Source type per trace (1-8, from TRID) */
    int    *orient;     /* Source orientation per trace (1-3, from TRID) */
    float  *x;          /* Horizontal positions (physical coordinates) */
    float  *z;          /* Vertical positions (physical coordinates) */
    float  *wav;        /* Residual waveforms [nsrc * nt], column-major */
    /* Receiver interpolation mode in force when the data were RECORDED
     * (recPar.int_vx / int_vz).  applyAdjointSource must apply the
     * transpose of whatever getRecTimes.c actually did, so these must
     * mirror rec.int_vx / rec.int_vz.  0 = plain grid-point sample,
     * 2 = co-location onto the Txx/Tzz node and time level.  Modes 1 and
     * 3 have no transpose and readResidual rejects them.
     *
     * Set by readResidual from its recPar argument -- do NOT rely on a
     * caller to assign these.  They used to be the caller's job and every
     * caller's memset left them at 0, which silently backpropagated the
     * mode-0 transpose for mode-2 recordings. */
    int     int_vx;
    int     int_vz;
} adjSrcPar;


/* --- Parameter activity mask for selectable parameter updates --- */
typedef struct _paramMask {
    int update_p1;   /* 1 = active, 0 = frozen (lambda/Vp) */
    int update_p2;   /* 1 = active, 0 = frozen (mu/Vs) */
    int update_p3;   /* 1 = active, 0 = frozen (rho) */
} paramMask;

void applyParamMask(float *g, int nmodel, int nparam,
                    const paramMask *pmask);
void restoreFrozenParams(float *x, const float *x_frozen,
                         int nmodel, int nparam,
                         const paramMask *pmask);

/* --- updateModel.c: Model vector / FD coefficient bridge for FWI --- */
void recomputeFDcoefficients(modPar *mod, bndPar *bnd);
void extractModelVector(float *x, modPar *mod, bndPar *bnd, int param);
void injectModelVector(float *x, modPar *mod, bndPar *bnd, int param);
void extractGradientVector(float *g, float *grad1, float *grad2, float *grad3,
                           modPar *mod, bndPar *bnd, int param);
void perturbFDcoefficients(modPar *mod, bndPar *bnd,
                           float *dpert, int param,
                           float *delta_rox, float *delta_roz,
                           float *delta_l2m, float *delta_lam,
                           float *delta_mul);

/* --- born_vsrc.c: Virtual source injection for Born/Hessian-vector --- */
void inject_born_vsrc_vel(modPar *mod,
                          float *txx_fwd, float *tzz_fwd, float *txz_fwd,
                          float *delta_rox, float *delta_roz,
                          float *born_vx, float *born_vz);
void inject_born_vsrc_stress(modPar *mod, bndPar *bnd,
                             float *vx_fwd, float *vz_fwd,
                             float *delta_l2m, float *delta_lam,
                             float *delta_mul,
                             float *born_txx, float *born_tzz,
                             float *born_txz);

/* --- born_shot.c: Forward Born propagation (J * dm) --- */
int born_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec,
              int ixsrc, int izsrc, float **src_nwav,
              checkpointPar *chk,
              float *delta_rox, float *delta_roz,
              float *delta_l2m, float *delta_lam, float *delta_mul,
              const char *file_born,
              int ishot, int nshots, int fileno,
              int verbose);

/* Maximum number of data components in a comp= list */
#define MAX_COMP 8

/* --- energy_weights.c: Energy-norm data weighting W_r --- */
/* Q = W_r^T W_r = diag(rho*I, S) with S the elastic compliance, so that
 * J = 0.5*|W_r*dd|^2 = 0.5*dd^T Q dd is the energy density (J/m^3) of the
 * data residual.  Holds the medium properties sampled at each receiver;
 * built once from the initial model so Q is a constant metric. */
typedef struct _energyWeightPar {
    int    nrec;      /* number of receivers (must equal traces per component) */
    float *rho;       /* [nrec] density at receiver                            */
    float *lam;       /* [nrec] Lame lambda at receiver                        */
    float *muu;       /* [nrec] Lame mu at receiver                            */
} energyWeight;

energyWeight *energyWeightsBuild(modPar *mod, bndPar *bnd, recPar *rec, int verbose);
int  energyWeightIsFluid(float mu_r, float lam_r);
void energyWeightApply(int ncomp, const int *chan, const float *r, float *psi,
                       const energyWeight *ew, int irec,
                       int idx_txx, int idx_tzz);
void energyWeightsFree(energyWeight *ew);

/* --- energy_imaging.c: energy-norm angle filter on the gradient ---
 * Elastic form of Rocha, Tanushev & Sava (2016, GEOPHYSICS 81(4), S151-S163).
 * Accumulates the kinetic and potential halves of the elastic energy inner
 * product between the forward and adjoint wavefields,
 *     K = sum_t dt rho (v_fwd . psi_v),  P = sum_t dt (sig_fwd : S : psi_sig)
 * and uses them to build a dimensionless per-point scattering-angle weight.
 * In velocity-stress variables K-P selects transmission (tomographic) and
 * K+P selects specular reflection (migration) -- the opposite pairing from
 * the acoustic potential formulation; see energy_imaging.c for the derivation.
 * Any non-zero mode changes the gradient and so breaks the Taylor test, the
 * same way grad_taper and smooth_grad do. */
#define EI_OFF   0   /* disabled; accumulators are no-ops          */
#define EI_TOMO  1   /* weight by |K-P|: keep transmission         */
#define EI_MIGR  2   /* weight by |K+P|: keep specular reflections */
#define EI_QC    3   /* accumulate and report, gradient untouched  */

/* Diagnostic fields retrievable with energyImagingGet.  Codes 0-3 are the
 * originals; at EI_MODESEP they report the P channel and the MIXED weight,
 * so an existing caller that only knows 0-3 keeps working and shows the
 * quantity that is actually applied to g_mu and g_rho.  Codes 4-9 are the
 * per-mode breakdown and return 0 (nothing written) at the other gammas. */
#define EI_GET_KIN   0   /* K          (K_P at EI_MODESEP)                 */
#define EI_GET_POT   1   /* V          (V_P at EI_MODESEP)                 */
#define EI_GET_TOMO  2   /* a_tomo     (MIXED a_tomo at EI_MODESEP)        */
#define EI_GET_MIGR  3   /* a_migr     (MIXED a_migr at EI_MODESEP)        */
#define EI_GET_KIN_S    4   /* K_S                 -- EI_MODESEP only      */
#define EI_GET_POT_S    5   /* V_S                 -- EI_MODESEP only      */
#define EI_GET_TOMO_P   6   /* a_tomo, P channel   -- EI_MODESEP only      */
#define EI_GET_MIGR_P   7   /* a_migr, P channel   -- EI_MODESEP only      */
#define EI_GET_TOMO_S   8   /* a_tomo, S channel   -- EI_MODESEP only      */
#define EI_GET_MIGR_S   9   /* a_migr, S channel   -- EI_MODESEP only      */
#define EI_GET_N       10

/* Which member of the one-parameter family of potential (strain) energy
 * pairings to use.  The members are identical for a single wavefield -- they
 * differ by the null Lagrangian N = div.div~ - d_i v_j d_j v~_i, whose
 * integral is a boundary term -- but NOT for the pointwise product of two
 * different fields, which is exactly what an imaging condition forms.
 *
 *   V_gamma = (l+2m) div.div~ + m curl.curl~ - gamma*m*N
 *
 * EI_GAMMA2 is the strain-energy pairing eps:C:eps~ and is what this code did
 * before the family was parameterised; EI_GAMMA0 is the divergence-curl
 * (Navier) form.  See energy_imaging.c section 2c for why they differ and
 * tests/gamma_energy/ for the plane-wave proof. */
#define EI_GAMMA2  2   /* eps:C:eps~ pairing (default, historical behaviour) */
#define EI_GAMMA0  0   /* (l+2m) div.div~ + m curl.curl~ (Navier form)       */

/* EI_MODESEP is NOT a member of the gamma family -- it occupies the same
 * switch because it answers the same question, but it is a different object.
 * Instead of one elastic energy pairing it applies Rocha's ACOUSTIC energy
 * norm twice, once to each Helmholtz scalar:
 *
 *     D = div v   obeys the scalar wave equation at Vp
 *     C = curl v  obeys the scalar wave equation at Vs
 *
 *     K_P = sum_t dt Ddot_f Ddot_a     V_P = sum_t dt Vp^2 grad D_f . grad D_a
 *     K_S = sum_t dt Cdot_f Cdot_a     V_S = sum_t dt Vs^2 grad C_f . grad C_a
 *
 * Because each channel uses its OWN mode velocity, Rocha's |k| = w/v holds by
 * construction and the plane-wave result is K+-V ~ (1 +- cos psi) with no
 * velocity ratio and no isotropic modulation -- the (l -+ 2m cos2theta) factor
 * of EI_GAMMA2 and the kappa dependence of EI_GAMMA0 are both gone exactly,
 * for both modes.  See energy_imaging.c section 2e.
 *
 * Cost: one extra scalar field and one extra gradient per wavefield per step.
 * Blind spot: a converted (P-S) pair contributes to NEITHER channel, so the
 * mask there is set by whatever unconverted energy is present. */
#define EI_MODESEP (-1) /* per-mode acoustic energy norm (D at Vp, C at Vs) */

void energyImagingInit(modPar *mod, int mode, int gamma, float alpha,
                       int verbose);
/* Accumulator called where psi_SIGMA is the valid multiplier (before
 * callAdjKernel).  Which half of the energy it feeds depends on gamma:
 * gamma=2 -> potential, gamma=0 -> kinetic.  Hence the call-site name. */
void energyImagingAccumAtSigma(modPar *mod, bndPar *bnd,
                               float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                               wflPar *wfl_adj, float dt);
/* Accumulator called where psi_V is the valid multiplier (after
 * callAdjKernel).  gamma=2 -> kinetic, gamma=0 -> potential. */
void energyImagingAccumAtVel(modPar *mod, bndPar *bnd,
                             float *fwd_vx, float *fwd_vz,
                             wflPar *wfl_adj, float dt);
void energyImagingApply(modPar *mod, bndPar *bnd,
                        float *grad_lam, float *grad_muu, float *grad_rho,
                        int verbose);
int  energyImagingGet(modPar *mod, bndPar *bnd, int which, float *out);
void energyImagingFree(void);

/* --- kh_misfit.c: Kirchhoff-Helmholtz (energy-flux) objective ---
 * J = sum_r w_r sum_n r_r^n with r = (dsig_ij n_j) dv_i the instantaneous
 * energy-flux density of the residual wavefield through the recording
 * surface.  Holds the receiver quadrature weights w_r and the surface
 * normal; built once from the receiver geometry.  Requires data_weight=0:
 * any velocity-vs-stress rescaling destroys the flux interpretation. */
typedef struct _khPar {
    int    nrec;      /* number of receivers (must equal traces per component) */
    float *w;         /* [nrec] receiver quadrature weight w_r (m)             */
    float  nz;        /* recording-surface normal, z component                 */
} khPar;

khPar *khWeightsBuild(recPar *rec, float normal_z, int verbose);
void   khWeightsFree(khPar *kh);
void   khResolveChannels(int ncomp, const int *chan, int *idx_vx, int *idx_vz,
                         int *idx_txz, int *idx_tzz, int *idx_txx);
double khApply(int ncomp, float *psi, const float *r, const khPar *kh,
               int irec, int idx_vx, int idx_vz, int idx_txz, int idx_tzz,
               double *flux_shear, double *flux_normal);

/* --- hess_shot.c: Gauss-Newton Hessian-vector product (J^T J dm) --- */
int hess_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec,
              int ixsrc, int izsrc, float **src_nwav,
              checkpointPar *chk,
              float *delta_rox, float *delta_roz,
              float *delta_l2m, float *delta_lam, float *delta_mul,
              const char *file_born, const char *comp_str,
              int ishot, int nshots, int fileno,
              int res_taper,
              float *hd1, float *hd2, float *hd3,
              int param, const float *comp_weights,
              const energyWeight *ew, int verbose);

/* --- bandpass_filter.c: Multiscale FWI frequency band filtering --- */
#define MAX_BANDS 20
typedef struct _bandPar {
    int    nbands;                    /* Number of frequency bands (0 = disabled) */
    float  flo[MAX_BANDS];            /* Low corner freq per band (Hz) */
    float  fhi[MAX_BANDS];            /* High corner freq per band (Hz) */
    int    niter_per_band[MAX_BANDS]; /* Max optimizer iterations per band */
} bandPar;
void bandpass_filter_sufile(const char *filename, float flo, float fhi);
void bandpass_filter_syndata(const char *file_rcv, const char *comp_str,
                             int fileno, float flo, float fhi);
void bandpass_filter_obsdata(const char *file_obs, const char *comp_str,
                             int nshots, float flo, float fhi,
                             int iband, int mpi_rank);
void bandpass_cleanup_obsbackups(const char *file_obs, const char *comp_str,
                                 int nshots);

/* --- computeResidual.c: Misfit and adjoint source --- */
float computeDataRMS(const char *filename);
float computeResidual(int ncomp, const char **obs_files, const char **syn_files,
                      const char *res_file, misfitType mtype,
                      const float *comp_weights, const energyWeight *ew,
                      const khPar *kh, int verbose);

/* --- aperture_taper.c: Geometric taper on the adjoint injection ---
 * Ramps the injection amplitude to zero over the outermost aper_taper
 * grid points of the receiver array, removing the array-truncation
 * diffractions that streak the gradient.  Applied outside the misfit,
 * so it is a search-direction weight, not part of J; see the comment
 * block in aperture_taper.c before relying on the gradient. */
void applyApertureTaper(adjSrcPar *adj, float dx, int aper_taper, int verbose);

/* NOTE: applyCosineTaper (residual_taper.c), buildGradTaperMask /
 * applyGradTaperMask (grad_taper_mask.c) and computeHydrophone
 * (computeHydrophone.c) are deliberately NOT declared here yet.  Four
 * drivers -- fwi_inversion.c, fwi_driver.c, hess_shot.c and
 * test_gradient_visual.c -- still carry their own file-local `static`
 * copies, and a `static` definition following a non-static declaration is
 * a hard error in C.  Declaring them here is the last step of that
 * de-duplication, once those four are switched over; until then new code
 * prototypes them locally.  The duplication is not hypothetical: the
 * copies drifted, and the Born path shipped without a _rp at all. */

/* --- scaling.c: Parameter scaling (0=none, 1=range normalization) --- */
void scaling_compute_m0_from_bounds(const float *m_min, const float *m_max,
                                    int nparam, float *m0, float *m_shift);
void scaling_normalize(float *v, int nmodel, int nparam,
                       const float *m0, const float *m_shift);
void scaling_denormalize(float *v, int nmodel, int nparam,
                         const float *m0, const float *m_shift);
void scaling_scale_gradient(float *g, int nmodel, int nparam, const float *m0);
void scaling_scale_hessian_vec(float *Hd, int nmodel, int nparam, const float *m0);
void scaling_denorm_direction(float *d, int nmodel, int nparam, const float *m0);

/* --- taper_source.c: Per-shot radial source taper --- */
float *buildSourceTaperMask(int nax, int naz, int sx, int sz,
                            float radius, float dx, int filtsize);
void applySourceTaper(float *arr, const float *mask, size_t sizem);

/* --- precond_shot.c: Receiver wavefield backprop for P4 preconditioner --- */
int precond_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                 recPar *rec,
                 int ixsrc, int izsrc, float **src_nwav,
                 checkpointPar *chk,
                 float *hess_lam, float *hess_muu, float *hess_rho,
                 float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
                 int verbose,
                 const char *syn_base, int fileno, const char *comp_str);

/* --- fwi_gradient.c: P4 Hessian cross-correlation --- */
void accumHessianP4(modPar *mod, bndPar *bnd,
                    float *fwd_vx, float *fwd_vz,
                    float *fwd_vx_prev, float *fwd_vz_prev,
                    float *rcv_vx, float *rcv_vz,
                    float *rcv_vx_prev, float *rcv_vz_prev,
                    float dt,
                    float *hess_lam, float *hess_muu, float *hess_rho,
                    float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho);

/* --- smooth2d.c: 2D Gaussian smoothing --- */
void smooth2d(float *data, int nx, int nz, int smooth_x, int smooth_z);
void smooth2d_padded(float *data, int nax, int naz,
                     int nx, int nz, int ibndx, int ibndz,
                     int smooth_x, int smooth_z);

/* --- preconditioner.c: Shin/Métivier diagonal pseudo-Hessian preconditioner --- */
void buildBlockPrecond(
    float *hess_lam, float *hess_muu, float *hess_rho,
    float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
    modPar *mod, bndPar *bnd,
    int param, int elastic,
    float precond_eps, float s1, float s2, float s3,
    float *P11, float *P12, float *P13,
    float *P22, float *P23, float *P33,
    int nmodel, int mpi_rank,
    float blend_depth, float taper_alpha,
    float *wfld_energy,
    float xmin, float xmax,
    int precond_mode);
void applyBlockPrecond(
    float *out, const float *in,
    const float *P11, const float *P12, const float *P13,
    const float *P22, const float *P23, const float *P33,
    int nmodel, int nparam, int norm_preserve);
float getPrecondNu(void);

/* --- fwi_gradient.c: Gradient kernels --- */
void accumGradient(modPar *mod, bndPar *bnd,
                   float *fwd_vx, float *fwd_vz,
                   float *fwd_vx_prev, float *fwd_vz_prev,
                   float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                   wflPar *wfl_adj, float dt,
                   float *grad_lam, float *grad_muu, float *grad_rho,
                   float *hess_lam, float *hess_muu, float *hess_rho,
                   float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
                   float *K_lam_tmp, float *K_muu_tmp, float *K_rho_tmp,
                   float *wfld_energy);
void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho, size_t sizem);
void accumGradient_rho_Dsig(modPar *mod, bndPar *bnd,
                            float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                            wflPar *wfl_adj,
                            float dt,
                            float *grad_rho);

/* --- acoustic4_adj.c: Acoustic discrete adjoint kernel --- */
int acoustic4_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
    float *vx, float *vz, float *p,
    float *rox, float *roz, float *l2m,
    int rec_delay, int rec_skipdt, int verbose);

/* --- acoustic_gradient.c: Acoustic gradient imaging condition --- */
void accumGradientAcoustic(modPar *mod, bndPar *bnd,
    float *fwd_vx, float *fwd_vz,
    float *fwd_vx_prev, float *fwd_vz_prev,
    wflPar *wfl_adj, float dt,
    float *grad_l2m, float *grad_rho,
    float *hess_l2m, float *hess_rho,
    float *wfld_energy);
void convertGradientAcoustic(float *grad_l2m, float *grad_rho,
    float *cp, float *rho, size_t sizem);

/* --- toy2dac_scaling.c: TOY2DAC first-iteration scaling --- */
float compute_toy2dac_scalingfactor(float *model_vec, float *grad_vec,
    int nvec, int verbose);
void apply_toy2dac_scaling(float *fcost, float *grad, int nvec, float sf);

/* --- source_estimation.c: cc1 source amplitude estimation --- */
float compute_cc1_shot(const char *syn_file, const char *obs_file,
    int verbose);
float compute_cc1_global(const char **syn_files, const char **obs_files,
    int nshots, int verbose);

/* --- regularization.c: Tikhonov + prior model --- */
float tikhonov_cost(float *model, int nx, int nz, float dx,
    float lambda, float lambda_x, float lambda_z,
    const int *ibathy, int npar);
void tikhonov_gradient(float *grad, float *model, int nx, int nz, float dx,
    float lambda, float lambda_x, float lambda_z,
    const int *ibathy, int npar);
float prior_cost(float *model, float *prior, int n, float alpha);
void prior_gradient(float *grad, float *model, float *prior, int n,
    float alpha);

/* --- shin_precond_acoustic.c: Shin diagonal preconditioner --- */
void shin_apply_precond(float *grad_preco, const float *grad,
    const float *hess_diag, int n, float eps_thresh,
    float *norm_ratio_out);

/* --- lag_gradient.c: time-lag extended imaging condition ---
 * Evaluates the lambda/mu gradient kernels at non-zero time lag,
 *   g(x,tau) = dt sum_t A(x,t) B(x,t+tau),
 * with A the forward strain and B the adjoint stress.  The accumulators
 * carry lambda and mu; the OUTPUT is g_Vp and g_Vs, converted per lag by the
 * same pointwise chain rule convertGradientToVelocity uses on the ordinary
 * gradient (that map is lag-independent, so it commutes with the lag
 * reconstruction).  There is no g_rho cube.  The adjoint source
 * is the data residual, so this is a lag-extension of dJ/dm and the tau=0
 * slice is the ordinary gradient (band limited to the frequencies
 * requested).
 *
 * Implemented as a product of running DFTs, which is what makes it fit
 * adj_shot's checkpoint scheme: the forward and adjoint states are only
 * ever co-resident at the same time index, and this formulation never
 * needs two different times at once.  Memory scales with nfreq, not with
 * the number of lags; the lag axis is chosen at write time. */
void lagGradientInit(modPar *mod, int nfreq, float fmin, float fmax,
    int verbose);
/* Fold one time step into the running DFTs.  Called from the backward
 * loop of adj_shot where psi_sigma is the valid multiplier (before
 * callAdjKernel), with the forward velocities of the same time index.
 * A no-op unless lagGradientInit was called with nfreq > 0. */
void lagGradientAccum(modPar *mod, bndPar *bnd,
    float *fwd_vx, float *fwd_vz,
    wflPar *wfl_adj, int it, float dt);
/* Reconstruct and write <base>_shotNNN_lag{vp,vs}.su plus a lagaxis.txt
 * description, and add into the shot stack.  nlag is forced odd so tau=0
 * is sampled; dlag is in seconds and need not be a multiple of mod->dt.
 * mask is an optional [nx*nz] interior-grid multiplier (ix*nz+iz) applied
 * to every lag -- pass the grad_taper mask, since adj_shot's taper acts on
 * the gradient and never reaches these accumulators. */
void lagGradientWrite(modPar *mod, bndPar *bnd, const char *base, int ishot,
    int nlag, float dlag, float *mask, int verbose);
/* Write the sum over all shots as <base>_lag{vp,vs}.su.  That stack, not
 * any single shot, is what the conventional gradient corresponds to. */
void lagGradientWriteStack(modPar *mod, const char *base, int verbose);
/* Release everything including the stack; call after the shot loop. */
void lagGradientFree(void);


/* --- space_lag_gradient.c: space-lag extended imaging condition ---
 * The same lambda/mu kernels with the two wavefields separated by a space
 * lag VECTOR h = (hx, hz) instead of a time lag,
 *   g(x,h) = dt sum_t A(x-h,t) B(x+h,t),
 * symmetric, so the relative separation is 2h.  Output is g_Vp and g_Vs per
 * lag, same pointwise chain rule, same absence of a g_rho cube.
 *
 * The OPPOSITE implementation choice from lag_gradient.c, for the reason
 * stated there: both fields ARE co-resident at the same time index, so a
 * space lag needs no transform pair and is accumulated directly in the time
 * domain.  Hence no band limiting and no aliasing -- the h=0 slice is
 * accumGradient's arithmetic over accumGradient's bounds in accumGradient's
 * order, so it reproduces the conventional gradient to float rounding.  The
 * costs invert: memory is 2*nhx*nhz*sizem, the lag axis must be fixed before
 * the run, and lags are integer multiples of the grid spacing (the shift is
 * an index offset, not a phase rotation).
 *
 * The conjugate variable of h is k_A + k_B = (2 w cos(theta)/v) N: the
 * migration wavenumber, whose DIRECTION is the reflector normal and needs no
 * velocity.  That is why the lag has to be a vector.
 *
 * nhx/nhz are forced odd so h=0 is sampled; dhx/dhz are strides in GRID
 * POINTS.  (nhx,1) and (1,nhz) give both components of k_A+k_B at cost
 * nhx+nhz rather than nhx*nhz, which is enough wherever one normal dominates
 * per image point.  shiftx/shiftz are the CALIBRATION of the h axis: an extra
 * grid shift D on the adjoint field must translate the cube by D/2 in BOTH x
 * and h; use even D.  See the file header. */
void slagGradientInit(modPar *mod, int nhx, int nhz, int dhx, int dhz,
    int shiftx, int shiftz, int verbose);
/* Add one time step into every lag accumulator.  Same call site as
 * lagGradientAccum (psi_sigma before callAdjKernel), but with no `it`:
 * nothing here depends on absolute time.  A no-op unless slagGradientInit
 * was called with nhx*nhz > 0. */
void slagGradientAccum(modPar *mod, bndPar *bnd,
    float *fwd_vx, float *fwd_vz,
    wflPar *wfl_adj, float dt);
/* Write <base>_shotNNN_slag{vp,vs}.su plus a slagaxis.txt description, and
 * add into the shot stack.  Cubes are nz samples by (nx*nhx*nhz) traces,
 * trace = (ihz*nhx + ihx)*nx + ix.  mask as for lagGradientWrite. */
void slagGradientWrite(modPar *mod, bndPar *bnd, const char *base, int ishot,
    float *mask, int verbose);
/* Write the shot sum as <base>_slag{vp,vs}.su -- the cube the conventional
 * gradient corresponds to. */
void slagGradientWriteStack(modPar *mod, const char *base, int verbose);
/* Pointer to the h=0 plane of the shot stack (ivar 0 = Vp, 1 = Vs), or NULL.
 * Lets the driver make the h=0 check numerically in the same run rather than
 * reading its own output back. */
float *slagGradientStackZeroLag(int ivar);
/* Release everything including the stack; call after the shot loop. */
void slagGradientFree(void);


/* ------------------------------------------------------------------ *
 * lag_cone_filter.c -- 3D slope (cone) filter on a lag cube.
 *
 * Splits g(tau,x,z) by the local slope |p| = |k|/|nu|, which is the
 * lag-domain measure of the opening angle: a SMALL slope is a nearly
 * lag-independent event, i.e. a wide angle, i.e. the TOMOGRAPHIC kernel; a
 * large slope is the specular reflection imprint.  C port of the validated
 * demo/gradient/lag_slope_filter.py:slope_filter3d.
 *
 * The filter is a global convolution, so it does NOT commute with the
 * pointwise-but-spatially-varying Lame->velocity chain rule.  Apply it to
 * the cubes lag_gradient.c already converted to (Vp, Vs), and do not
 * convert again downstream.
 * ------------------------------------------------------------------ */
typedef struct {
	float s_pass;      /* ms/m, H = 1 at or below                    */
	float s_stop;      /* ms/m, H = 0 at or above                    */
	float tap_tau;     /* raised-cosine edge fraction on the lag axis */
	float tap_x;       /* ... on x; NOTE this attenuates the gradient */
	float tap_z;       /* ... on z;     near the model edges          */
	int   pad;         /* minimum zero-pad factor per axis (>= 1)     */
	int   nu0_keep;    /* 0 = 'limit' (default), 1 = pass the nu=0 plane */
} lagConePar;

typedef struct {
	int   nlag, nx, nz;          /* input shape                        */
	int   Nt, Nx, Nz;            /* padded transform lengths actually used */
	float nu_max;                /* Nyquist on the lag axis, Hz        */
	float energy_frac_nu0;       /* share of the spectrum at nu = 0    */
	float energy_frac_passed;    /* share of the spectrum kept         */
} lagConeStats;

/* Reference defaults: s_pass=0.20, s_stop=0.30 ms/m, 10% tapers, pad=2. */
void  lagConeDefaults(lagConePar *p);
/* H(s): 1 below s_pass, cos^2 transition, 0 above s_stop. */
float lagConeSlopeResponse(float s, float s_pass, float s_stop);
/* Full opening angle 2theta [deg] -> slope |p| = cos(theta)/v [ms/m].  The
 * only place a velocity enters, and only to turn an angle into a slope. */
float lagConeAngleToSlope(float theta_deg, float v);
/* Filter one cube, order (lag, x, z) with z fastest.  `tapered` may be NULL,
 * but the caller needs it to form the exact complement
 * rejected = tapered - passed; comparing against the RAW input would credit
 * the edge taper to the filter. */
void  lagConeFilter(const float *cube, int nlag, int nx, int nz,
    float dx, float dz, float dtau, const lagConePar *p,
    float *passed, float *tapered, lagConeStats *st, int verbose);
/* Verify the FFT round-trip normalisation, the only backend-dependent part.
 * Returns max error relative to input RMS; > 1e-5 means trouble. */
float lagConeSelfTest(int verbose);

/* --- lag_gradient.c, declared here because it needs lagConePar ---
 * Build this shot's (Vp, Vs) lag cubes, cone-filter each one, and ADD the
 * tau=0 slice of the passed part into gvp/gvs ([nx*nz], ix*nz+iz, interior
 * grid).  That slice is the shot's tomographic gradient contribution,
 * already in velocity: the filter does not commute with the chain rule, so
 * it must not be converted again downstream.  There is no rho output --
 * freeze rho when inverting with this. */
void lagGradientTomo(modPar *mod, bndPar *bnd, int nlag, float dlag,
    float *mask, const lagConePar *p, float *gvp, float *gvs, int verbose);


#if __STDC_VERSION__ >= 199901L
  /* "restrict" is a keyword */
#else
#define restrict
#endif

#endif /* FDELFWI_H */

