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
    MISFIT_CORRELATION       /* Cross-correlation (TODO) */
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
} adjSrcPar;


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
              int param, int verbose);

/* --- fwi_gradient.c: Gradient kernels --- */
void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho, size_t sizem);
void accumGradient_rho_Dsig(modPar *mod, bndPar *bnd,
                            float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                            wflPar *wfl_adj,
                            float dt,
                            float *grad_rho);


#if __STDC_VERSION__ >= 199901L
  /* "restrict" is a keyword */
#else
#define restrict
#endif

