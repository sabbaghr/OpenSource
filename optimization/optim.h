/*
 * optim.h - C translation of SEISCOPE Optimization Toolbox interface.
 *
 * Provides L-BFGS, PLBFGS, PNLCG, TRN, and Steepest Descent with
 * Wolfe linesearch and optional box constraints. Uses reverse
 * communication: the caller provides cost and gradient when requested
 * via optFlag.
 *
 * Reference: SEISCOPE Optimization Toolbox (Metivier & Brossier, 2016)
 *            Nocedal & Wright, "Numerical Optimization", 2006
 *
 * Original Fortran: TOOLBOX_OPTIMIZATION (BSD license, SEISCOPE II 2013-2016)
 */

#ifndef OPTIM_H
#define OPTIM_H

#include <stdlib.h>

/* Reverse communication flags (replaces Fortran CHARACTER*4 FLAG) */
typedef enum {
	OPT_INIT,   /* First call: initialize optimizer */
	OPT_GRAD,   /* User must compute cost and gradient at current x */
	OPT_CONV,   /* Convergence reached */
	OPT_FAIL,   /* Linesearch failure */
	OPT_NSTE,   /* New step accepted (iteration complete) */
	OPT_HESS,   /* TRN: user must compute Hd = H * d (Hessian-vector product) */
	OPT_PREC    /* PLBFGS: user must apply preconditioner to q_plb */
} optFlag;

/* Algorithm selector */
typedef enum {
	ALG_SD,      /* Steepest descent */
	ALG_LBFGS,   /* Limited-memory BFGS */
	ALG_PLBFGS,  /* Preconditioned L-BFGS */
	ALG_PNLCG,  /* Preconditioned nonlinear conjugate gradient (Dai-Yuan) */
	ALG_TRN,     /* Truncated Newton */
	ALG_ENRICHED /* Enriched method (L-BFGS + TN hybrid, Morales & Nocedal 2002) */
} optAlg;

/* Internal linesearch state (replaces Fortran CHARACTER*8 task) */
typedef enum {
	LS_NEW_GRAD,  /* Need new cost/gradient evaluation */
	LS_NEW_STEP,  /* Step accepted */
	LS_FAILURE    /* Linesearch failed */
} lsTask;

/* TRN internal communication state */
typedef enum {
	TRN_DESC,    /* Computing descent direction (inner CG) */
	TRN_NSTE     /* Linesearch for new step */
} trnComm;

/* TRN CG phase */
typedef enum {
	CG_INIT,     /* CG initialization */
	CG_IRUN      /* CG running */
} cgPhase;

/* Optimizer state structure (replaces Fortran optim_type) */
typedef struct {
	/* Algorithm selection */
	optAlg  algorithm;
	int     debug;
	int     print_flag;     /* 1=write convergence file */

	/* Dimensions */
	int     n;              /* model vector length (set by init) */

	/* Iteration control */
	int     niter_max;      /* max optimization iterations */
	int     cpt_iter;       /* current iteration counter */
	float   conv;           /* convergence tolerance: fcost/f0 < conv */
	float   f0;             /* initial cost (reference) */

	/* Linesearch state */
	int     first_ls;       /* 1=initialization step needed */
	lsTask  ls_task;        /* internal linesearch state */
	int     nls_max;        /* max linesearch iterations (default 20) */
	int     cpt_ls;         /* linesearch iteration counter */
	int     nfwd_pb;        /* forward problem counter */
	float   fk;             /* cost at start of current linesearch */
	float   m1;             /* Wolfe sufficient decrease (default 1e-4) */
	float   m2;             /* Wolfe curvature (default 0.9) */
	float   mult_factor;    /* bracket expansion factor (default 10) */
	float   alpha_L;        /* left bracket for step size */
	float   alpha_R;        /* right bracket (0 = unbounded) */
	float   alpha;          /* current step size */
	float   q0;             /* initial directional derivative */
	float   q;              /* current directional derivative */

	/* Work arrays (size n, allocated by init) */
	float  *xk;             /* saved iterate for linesearch rollback */
	float  *grad;           /* saved gradient */
	float  *descent;        /* descent direction */

	/* L-BFGS / PLBFGS history (flat n*l arrays, column-major) */
	int     l;              /* max history pairs (user sets before INIT) */
	int     cpt_lbfgs;      /* current history pair counter */
	float  *sk;             /* step differences [n*l] */
	float  *yk;             /* gradient differences [n*l] */

	/* PLBFGS-specific (preconditioned L-BFGS) */
	float  *q_plb;          /* work vector for preconditioning [n] */
	float  *alpha_plb;      /* first-loop alphas [cpt_lbfgs] */
	float  *rho_plb;        /* first-loop rhos [cpt_lbfgs] */

	/* PNLCG-specific (Dai-Yuan CG) */
	float  *grad_prev;      /* previous gradient [n] */
	float  *descent_prev;   /* previous descent direction [n] */

	/* TRN-specific (Truncated Newton) */
	trnComm comm;           /* TRN communication state */
	cgPhase CG_phase;       /* CG phase (INIT/IRUN) */
	int     conv_CG;        /* CG convergence flag */
	int     niter_max_CG;   /* max CG iterations per outer iteration */
	int     cpt_iter_CG;    /* CG iteration counter */
	int     nhess;          /* Hessian-vector product counter */
	float   eta;            /* Eisenstat-Walker forcing term */
	float   norm_grad;      /* current gradient norm */
	float   norm_grad_m1;   /* previous gradient norm */
	float   norm_residual;  /* CG residual norm */
	float   qk_CG;          /* CG quadratic model value */
	float   qkm1_CG;        /* previous CG quadratic model value */
	float   hessian_term;   /* accumulated Hessian term for qk_CG */
	float  *residual;       /* CG residual [n] */
	float  *d;              /* CG direction [n] */
	float  *Hd;             /* Hessian-vector product result [n] */
	float  *eisenvect;      /* work vector for forcing term [n] */
	/* descent_prev is shared with PNLCG */

	/* Enriched method state (Morales & Nocedal 2002) */
	int     enr_method;     /* 0=L-BFGS phase, 1=HFN (TN) phase */
	int     enr_l;          /* L-BFGS cycle length (dynamically adjusted) */
	int     enr_t;          /* HFN cycle length (dynamically adjusted) */
	int     enr_k;          /* step counter within current cycle */
	int     enr_profit;     /* consecutive profitable HFN steps */
	int     enr_force2;     /* force at least 2 HFN iterations */
	int     enr_first;      /* first time entering HFN (startup: start with TRN) */
	int     enr_maxcg;      /* max CG iterations for current HFN cycle */
	trnComm enr_comm;       /* internal comm state (DESC=CG or NSTE=linesearch) */
	float  *enr_precond_z;  /* preconditioner workspace [n] */

	/* Bound constraints */
	int     bound;          /* 0=off, 1=on */
	float   threshold;      /* bound tolerance (default 0) */
	float  *lb;             /* lower bounds [n] (user-allocated if bound=1) */
	float  *ub;             /* upper bounds [n] (user-allocated if bound=1) */
} optim_type;


/* ================================================================
 * Public API
 * ================================================================ */

/* L-BFGS optimizer (reverse communication). */
void lbfgs_run(int n, float *x, float fcost, float *grad,
               optim_type *opt, optFlag *flag);

/* Preconditioned L-BFGS (reverse communication).
 * Returns OPT_PREC when user must apply preconditioner to opt->q_plb.
 * grad_preco: preconditioned gradient (used only at first iteration). */
void plbfgs_run(int n, float *x, float fcost, float *grad,
                float *grad_preco, optim_type *opt, optFlag *flag);

/* Preconditioned nonlinear conjugate gradient - Dai-Yuan (reverse communication).
 * grad_preco: preconditioned gradient at current x. */
void pnlcg_run(int n, float *x, float fcost, float *grad,
               float *grad_preco, optim_type *opt, optFlag *flag);

/* Truncated Newton (reverse communication).
 * Returns OPT_HESS when user must compute opt->Hd = H * opt->d. */
void trn_run(int n, float *x, float fcost, float *grad,
             optim_type *opt, optFlag *flag);

/* Steepest descent optimizer (reverse communication, same interface). */
void steepest_descent_run(int n, float *x, float fcost, float *grad,
                          optim_type *opt, optFlag *flag);

/* Enriched method: L-BFGS + TN hybrid (reverse communication).
 * Starts with TN to seed L-BFGS buffer, switches to L-BFGS as workhorse,
 * dynamically switches back to TN when L-BFGS stalls.
 * Returns OPT_HESS when Hessian-vector product is needed (same as TRN). */
void enriched_run(int n, float *x, float fcost, float *grad,
                  optim_type *opt, optFlag *flag);

/* Free all internally allocated arrays. Called automatically on CONV/FAIL,
 * but can be called manually for early termination. */
void optim_finalize(optim_type *opt);


/* ================================================================
 * Shared utility functions (also usable standalone)
 * ================================================================ */

/* L2 norm: ||x||_2 */
float optim_norm_l2(int n, const float *x);

/* Dot product: <x, y> */
float optim_dot(int n, const float *x, const float *y);

/* Project x into box [lb, ub] with threshold offset */
void optim_project(int n, optim_type *opt, float *x);

/* Wolfe linesearch (called internally by all optimizers) */
void optim_wolfe_linesearch(int n, float *x, float fcost, float *grad,
                            optim_type *opt);

/* Convergence test: returns 1 if fcost/f0 < conv or cpt_iter >= niter_max */
int optim_test_conv(optim_type *opt, float fcost);

/* Write convergence info to iterate_XX.dat file */
void optim_print_info(int n, const char *tag, optim_type *opt,
                      float fcost, optFlag flag);

/* L-BFGS history management (shared between lbfgs, plbfgs, enriched) */
void optim_save_lbfgs(int n, optim_type *opt, float *x, float *grad);
void optim_update_lbfgs(int n, optim_type *opt, float *x, float *grad);

/* L-BFGS two-loop recursion: output = -H(m) * input.
 * Used by lbfgs for descent direction and by enriched as CG preconditioner. */
void optim_lbfgs_apply(int n, optim_type *opt, const float *input,
                        float *output);

/* TRN-specific print info */
void print_info_trn(int n, optim_type *opt, float fcost, optFlag flag);

#endif /* OPTIM_H */
