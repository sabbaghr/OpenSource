/*
 * enriched.c - Enriched optimization: L-BFGS + Truncated Newton hybrid.
 *
 * Implements the Enriched Method (Morales & Nocedal 2002, Metivier 2017):
 *   1. Starts with TRN to seed L-BFGS buffer with true Hessian curvature
 *   2. Switches to L-BFGS as workhorse (cheap iterations, good curvature)
 *   3. Switches back to TRN (preconditioned by L-BFGS) when L-BFGS stalls
 *   4. ADJUST dynamically controls cycle lengths based on step quality
 *
 * Inner CG pairs (d, Hd) are stored in L-BFGS buffer for curvature seeding.
 * TRN inner CG is preconditioned by optim_lbfgs_apply() (L-BFGS two-loop).
 *
 * Reference: Morales & Nocedal, Comput. Optim. Appl., 21, 143-154, 2002
 *            Metivier et al., SIAM Review, 59(1), 153-195, 2017
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "optim.h"


/* File handle for enriched convergence history */
static FILE *_enr_fp = NULL;


/*--------------------------------------------------------------------
 * save_lbfgs_pair -- Store a (s,y) pair directly into L-BFGS buffer.
 *
 * Used for curvature seeding: stores inner CG pairs (d, Hd) where
 * d^T Hd > 0 satisfies the secant condition.
 *
 * If buffer not full: store at cpt_lbfgs-1, increment counter.
 * If buffer full: shift left, store at end.
 *--------------------------------------------------------------------*/
static void save_lbfgs_pair(int n, optim_type *opt,
                            const float *s_vec, const float *y_vec)
{
	int j;

	if (opt->cpt_lbfgs <= opt->l) {
		j = opt->cpt_lbfgs - 1;
		memcpy(&opt->sk[j * n], s_vec, n * sizeof(float));
		memcpy(&opt->yk[j * n], y_vec, n * sizeof(float));
		opt->cpt_lbfgs++;
	} else {
		/* Full: shift columns left, store new at end */
		for (j = 0; j < opt->l - 1; j++) {
			memcpy(&opt->sk[j * n], &opt->sk[(j + 1) * n],
			       n * sizeof(float));
			memcpy(&opt->yk[j * n], &opt->yk[(j + 1) * n],
			       n * sizeof(float));
		}
		j = opt->l - 1;
		memcpy(&opt->sk[j * n], s_vec, n * sizeof(float));
		memcpy(&opt->yk[j * n], y_vec, n * sizeof(float));
	}
}


/*--------------------------------------------------------------------
 * init_enriched -- Allocate arrays and initialize enriched state.
 *
 * Starts in HFN (TRN) mode to seed L-BFGS buffer with true
 * Hessian curvature from inner CG pairs.
 *--------------------------------------------------------------------*/
static void init_enriched(int n, float *x, float fcost, float *grad,
                          optim_type *opt)
{
	opt->n = n;

	/* Counters */
	opt->cpt_iter = 0;
	opt->f0 = fcost;
	opt->nfwd_pb = 0;
	opt->nhess = 0;

	opt->eta = 0.9f;

	/* Linesearch parameters */
	opt->m1 = 1e-4f;
	opt->m2 = 0.9f;
	opt->mult_factor = 10.0f;
	opt->fk = fcost;
	if (opt->nls_max <= 0) opt->nls_max = 20;
	opt->cpt_ls = 0;
	opt->first_ls = 1;
	if (opt->alpha <= 0.0f) opt->alpha = 1.0f;

	/* L-BFGS history arrays */
	opt->sk = (float *)calloc((size_t)n * opt->l, sizeof(float));
	opt->yk = (float *)calloc((size_t)n * opt->l, sizeof(float));
	opt->cpt_lbfgs = 1;

	/* Work arrays */
	opt->xk = (float *)malloc(n * sizeof(float));
	memcpy(opt->xk, x, n * sizeof(float));

	opt->grad = (float *)malloc(n * sizeof(float));
	memcpy(opt->grad, grad, n * sizeof(float));

	opt->descent = (float *)malloc(n * sizeof(float));
	opt->descent_prev = (float *)malloc(n * sizeof(float));
	opt->residual = (float *)malloc(n * sizeof(float));
	opt->d = (float *)malloc(n * sizeof(float));
	opt->Hd = (float *)malloc(n * sizeof(float));
	opt->eisenvect = (float *)malloc(n * sizeof(float));
	opt->enr_precond_z = (float *)malloc(n * sizeof(float));

	/* Gradient norm */
	opt->norm_grad = optim_norm_l2(n, grad);

	/* Enriched state: start in HFN mode (startup phase) */
	opt->enr_method = 1;          /* 1 = HFN */
	opt->enr_first = 1;           /* startup: first TRN phase */
	if (opt->enr_l <= 0) opt->enr_l = 20;  /* default L-BFGS cycle */
	opt->enr_t = 2;               /* initial TRN cycle length */
	opt->enr_k = 0;               /* step counter */
	opt->enr_profit = 0;
	opt->enr_force2 = 0;
	if (opt->enr_maxcg <= 0) opt->enr_maxcg = 5;
	opt->enr_comm = TRN_DESC;     /* start with CG descent */
}


/*--------------------------------------------------------------------
 * forcing_term_enriched -- Eisenstat-Walker forcing term (eta_k,1).
 *
 * eta = ||grad_new - residual|| / ||grad_prev||
 * Safeguard: if eta_prev^phi > 0.1, eta = max(eta, eta_prev^phi)
 *--------------------------------------------------------------------*/
static void forcing_term_enriched(int n, float *grad, optim_type *opt)
{
	int i;
	float eta_save, eta_save_power, norm_eisenvect;

	eta_save = opt->eta;

	for (i = 0; i < n; i++)
		opt->eisenvect[i] = grad[i] - opt->residual[i];

	norm_eisenvect = optim_norm_l2(n, opt->eisenvect);
	opt->eta = norm_eisenvect / opt->norm_grad_m1;

	/* Safeguard (Eisenstat & Walker) */
	eta_save_power = powf(eta_save, (1.0f + sqrtf(5.0f)) / 2.0f);
	if (eta_save_power > 0.1f)
		opt->eta = fmaxf(opt->eta, eta_save_power);

	if (opt->eta > 1.0f)
		opt->eta = 0.9f;
}


/*--------------------------------------------------------------------
 * adjust_enriched -- ADJUST procedure (Morales & Nocedal 2002).
 *
 * Dynamically switches between L-BFGS and HFN modes based on
 * step quality. Biased toward L-BFGS (long L-BFGS cycles,
 * short TRN cycles).
 *--------------------------------------------------------------------*/
static void adjust_enriched(optim_type *opt)
{
	opt->enr_k++;

	if (opt->enr_method == 0) {
		/* L-BFGS phase */
		if (opt->enr_k >= opt->enr_l) {
			/* L-BFGS cycle complete: switch to HFN */
			if (opt->enr_first) {
				opt->enr_maxcg = 5;
				opt->enr_t = 2;
				opt->enr_force2 = 0;
				opt->enr_first = 0;
			}
			opt->enr_method = 1;
			opt->enr_k = 0;
			opt->enr_profit = 0;
		}
	} else {
		/* HFN phase */
		if (opt->alpha >= 0.8f) {
			/* Profitable Newton step */
			opt->enr_profit++;
		} else {
			/* Unprofitable step */
			if (opt->enr_force2 && opt->enr_k == 1) {
				/* Give second chance */
				return;
			}
			opt->enr_t = (opt->enr_k - 1 > 2) ? opt->enr_k - 1 : 2;
			opt->enr_method = 0;
			opt->enr_k = 0;
			return;
		}

		if (opt->enr_k >= opt->enr_t) {
			/* HFN cycle complete */
			if (opt->enr_profit == opt->enr_k)
				opt->enr_t++;  /* all profitable: extend */
			opt->enr_force2 = (opt->enr_profit >= 2) ? 1 : 0;
			opt->enr_method = 0;
			opt->enr_k = 0;
		}
	}
}


/*--------------------------------------------------------------------
 * pcg_enriched -- Preconditioned CG inner loop for HFN mode.
 *
 * L-BFGS preconditioner: z = -H(m) * r  via optim_lbfgs_apply().
 * Stores (d, Hd) pairs in L-BFGS buffer for curvature seeding.
 *
 * Returns via opt->conv_CG:
 *   0 = need another Hessian-vector product (OPT_HESS)
 *   1 = CG converged or negative curvature (done)
 *--------------------------------------------------------------------*/
static void pcg_enriched(int n, float *grad, optim_type *opt,
                         optFlag *flag)
{
	int i;
	float dHd, alpha_cg, beta, rho_new, rho_old;

	if (opt->CG_phase == CG_INIT) {
		/* Initialize preconditioned CG */
		memcpy(opt->residual, grad, n * sizeof(float));

		/* z0 = -H(m) * r0  (L-BFGS preconditioner, returns -H*input) */
		optim_lbfgs_apply(n, opt, opt->residual, opt->enr_precond_z);
		/* enr_precond_z = -H(m)*residual, so this is -z0 in alg notation */
		/* We want d = -z0, which is what lbfgs_apply gives us */
		memcpy(opt->d, opt->enr_precond_z, n * sizeof(float));

		for (i = 0; i < n; i++) {
			opt->Hd[i] = 0.0f;
			opt->descent[i] = 0.0f;
		}

		/* rho_0 = <r0, z0> = <r0, -enr_precond_z> = -<r0, enr_precond_z> */
		/* Since enr_precond_z = -H*r = -z, we have <r, z> = -<r, enr_precond_z> */
		opt->norm_residual = optim_norm_l2(n, opt->residual);
		opt->conv_CG = 0;
		opt->cpt_iter_CG = 0;
		opt->CG_phase = CG_IRUN;

	} else {
		/* Process Hessian-vector product Hd, do one CG step */
		dHd = optim_dot(n, opt->d, opt->Hd);

		if (dHd <= 0.0f) {
			/* Negative curvature: stop CG */
			opt->conv_CG = 1;
			if (opt->cpt_iter_CG == 0) {
				/* First iteration: use preconditioned steepest descent */
				memcpy(opt->descent, opt->d, n * sizeof(float));
			}
			return;
		}

		/* Store (d, Hd) as curvature pair in L-BFGS buffer */
		{
			float norm_d = optim_norm_l2(n, opt->d);
			if (norm_d > 0.0f) {
				save_lbfgs_pair(n, opt, opt->d, opt->Hd);
			}
		}

		/* rho_old = <r, z> where z = -enr_precond_z */
		rho_old = -optim_dot(n, opt->residual, opt->enr_precond_z);

		/* CG update */
		alpha_cg = rho_old / dHd;

		memcpy(opt->descent_prev, opt->descent, n * sizeof(float));
		for (i = 0; i < n; i++) {
			opt->descent[i] += alpha_cg * opt->d[i];
			opt->residual[i] += alpha_cg * opt->Hd[i];
		}

		opt->cpt_iter_CG++;
		opt->norm_residual = optim_norm_l2(n, opt->residual);

		/* Check stopping criterion */
		if (opt->norm_residual <= opt->eta * opt->norm_grad ||
		    opt->cpt_iter_CG >= opt->enr_maxcg) {
			opt->conv_CG = 1;
			return;
		}

		/* Precondition new residual: z_new = -H(m) * r_new */
		optim_lbfgs_apply(n, opt, opt->residual, opt->enr_precond_z);

		/* rho_new = <r_new, z_new> = -<r_new, enr_precond_z> */
		rho_new = -optim_dot(n, opt->residual, opt->enr_precond_z);

		beta = rho_new / rho_old;

		/* d_new = -z_new + beta * d_old = enr_precond_z + beta * d */
		for (i = 0; i < n; i++)
			opt->d[i] = opt->enr_precond_z[i] + beta * opt->d[i];

		opt->conv_CG = 0;
	}
}


/*--------------------------------------------------------------------
 * print_info_enriched -- Write enriched convergence history.
 *
 * Output file: iterate_ENR.dat
 *--------------------------------------------------------------------*/
static void print_info_enriched(int n, optim_type *opt, float fcost,
                                optFlag flag)
{
	if (!opt->print_flag) return;

	if (flag == OPT_INIT) {
		_enr_fp = fopen("iterate_ENR.dat", "w");
		if (!_enr_fp) return;

		fprintf(_enr_fp,
			"**********************************************************************\n");
		fprintf(_enr_fp,
			"              ENRICHED OPTIMIZATION ALGORITHM\n");
		fprintf(_enr_fp,
			"          (L-BFGS + TN hybrid, Morales & Nocedal 2002)\n");
		fprintf(_enr_fp,
			"**********************************************************************\n");
		fprintf(_enr_fp, "     Convergence criterion  : %10.2e\n", opt->conv);
		fprintf(_enr_fp, "     Niter_max              : %7d\n", opt->niter_max);
		fprintf(_enr_fp, "     Initial cost is        : %10.2e\n", opt->f0);
		fprintf(_enr_fp, "     Initial norm_grad is   : %10.2e\n", opt->norm_grad);
		fprintf(_enr_fp, "     L-BFGS cycle length    : %7d\n", opt->enr_l);
		fprintf(_enr_fp, "     Memory parameter m     : %7d\n", opt->l);
		fprintf(_enr_fp, "     Initial maxcg          : %7d\n", opt->enr_maxcg);
		fprintf(_enr_fp,
			"**********************************************************************\n");
		fprintf(_enr_fp,
			"   Niter        fk      ||gk||      fk/f0"
			"         alpha  method   nls   nit_CG"
			"         eta    ngrad   nhess\n");
		fprintf(_enr_fp,
			"%6d%12.2e%12.2e%12.2e%12.2e    %s%8d%8d%12.2e%8d%8d\n",
			opt->cpt_iter, fcost, opt->norm_grad,
			1.0f, 0.0f,
			opt->enr_method ? "HFN" : " LB",
			opt->cpt_ls, opt->cpt_iter_CG,
			opt->eta, opt->nfwd_pb, opt->nhess);
		fflush(_enr_fp);

	} else if (flag == OPT_CONV) {
		if (!_enr_fp) return;
		fprintf(_enr_fp,
			"**********************************************************************\n");
		if (opt->cpt_iter >= opt->niter_max)
			fprintf(_enr_fp,
				"  STOP: MAXIMUM NUMBER OF ITERATION REACHED\n");
		else
			fprintf(_enr_fp,
				"  STOP: CONVERGENCE CRITERION SATISFIED\n");
		fprintf(_enr_fp,
			"**********************************************************************\n");
		fclose(_enr_fp);
		_enr_fp = NULL;

	} else if (flag == OPT_FAIL) {
		if (!_enr_fp) return;
		fprintf(_enr_fp,
			"**********************************************************************\n");
		fprintf(_enr_fp,
			"  STOP: LINESEARCH FAILURE\n");
		fprintf(_enr_fp,
			"**********************************************************************\n");
		fclose(_enr_fp);
		_enr_fp = NULL;

	} else {
		/* OPT_NSTE: write iteration row */
		if (!_enr_fp) return;
		fprintf(_enr_fp,
			"%6d%12.2e%12.2e%12.2e%12.2e    %s%8d%8d%12.2e%8d%8d\n",
			opt->cpt_iter, fcost, opt->norm_grad,
			(opt->f0 > 0.0f) ? fcost / opt->f0 : 0.0f,
			opt->alpha,
			opt->enr_method ? "HFN" : " LB",
			opt->cpt_ls, opt->cpt_iter_CG,
			opt->eta, opt->nfwd_pb, opt->nhess);
		fflush(_enr_fp);
	}
}


/*--------------------------------------------------------------------
 * finalize_enriched -- Free all enriched arrays.
 *--------------------------------------------------------------------*/
static void finalize_enriched(optim_type *opt)
{
	if (opt->sk) { free(opt->sk); opt->sk = NULL; }
	if (opt->yk) { free(opt->yk); opt->yk = NULL; }
	if (opt->xk) { free(opt->xk); opt->xk = NULL; }
	if (opt->grad) { free(opt->grad); opt->grad = NULL; }
	if (opt->descent) { free(opt->descent); opt->descent = NULL; }
	if (opt->descent_prev) { free(opt->descent_prev); opt->descent_prev = NULL; }
	if (opt->residual) { free(opt->residual); opt->residual = NULL; }
	if (opt->d) { free(opt->d); opt->d = NULL; }
	if (opt->Hd) { free(opt->Hd); opt->Hd = NULL; }
	if (opt->eisenvect) { free(opt->eisenvect); opt->eisenvect = NULL; }
	if (opt->enr_precond_z) { free(opt->enr_precond_z); opt->enr_precond_z = NULL; }
}


/*--------------------------------------------------------------------
 * enriched_run -- Main enriched reverse communication dispatcher.
 *
 * Interleaves L-BFGS and HFN (TRN) phases, sharing L-BFGS
 * curvature history. Starts in HFN mode to seed L-BFGS buffer.
 *
 * L-BFGS buffer management: enriched uses save_lbfgs_pair() for all
 * buffer writes (both CG curvature pairs and outer step pairs).
 * Does NOT use optim_save_lbfgs/optim_update_lbfgs since those
 * assume a save-then-difference pattern incompatible with direct
 * CG pair insertion.
 *
 * Returns:
 *   OPT_GRAD: compute cost and gradient at current x
 *   OPT_HESS: compute opt->Hd = H * opt->d
 *   OPT_NSTE: new step accepted
 *   OPT_CONV: converged
 *   OPT_FAIL: linesearch failed
 *--------------------------------------------------------------------*/
void enriched_run(int n, float *x, float fcost, float *grad,
                  optim_type *opt, optFlag *flag)
{
	int i;

	if (*flag == OPT_INIT) {
		/*--- Initialization ---*/
		init_enriched(n, x, fcost, grad, opt);
		print_info_enriched(n, opt, fcost, OPT_INIT);
		opt->nfwd_pb++;

		/* Start in HFN mode: begin CG */
		opt->enr_comm = TRN_DESC;
		opt->CG_phase = CG_INIT;
		opt->conv_CG = 0;

		/* Initialize CG (sets up d = -z0) */
		pcg_enriched(n, grad, opt, flag);
		/* Need first Hessian-vector product */
		*flag = OPT_HESS;
		opt->nhess++;
		return;
	}

	if (opt->enr_method == 1 && opt->enr_comm == TRN_DESC) {
		/*--- HFN mode: inner CG iteration ---*/
		pcg_enriched(n, grad, opt, flag);

		if (opt->conv_CG) {
			/* CG done: proceed to linesearch */
			opt->enr_comm = TRN_NSTE;
			opt->CG_phase = CG_INIT;
			optim_wolfe_linesearch(n, x, fcost, grad, opt);
			*flag = OPT_GRAD;
			opt->nfwd_pb++;
		} else {
			/* Need another Hessian-vector product */
			*flag = OPT_HESS;
			opt->nhess++;
		}
		return;
	}

	/*--- Linesearch phase (both L-BFGS and HFN) ---*/
	optim_wolfe_linesearch(n, x, fcost, grad, opt);

	if (opt->ls_task == LS_NEW_STEP) {
		/* Step accepted */
		int prev_method = opt->enr_method;

		opt->cpt_iter++;
		opt->norm_grad_m1 = opt->norm_grad;
		opt->norm_grad = optim_norm_l2(n, grad);

		/* Store outer (s,y) pair directly into L-BFGS buffer.
		 * s = x_new - x_old (xk saved by linesearch)
		 * y = grad_new - grad_old (opt->grad saved from previous step)
		 * Must compute BEFORE updating opt->grad. */
		{
			float *s_tmp = (float *)malloc(n * sizeof(float));
			float *y_tmp = (float *)malloc(n * sizeof(float));
			for (i = 0; i < n; i++) {
				s_tmp[i] = x[i] - opt->xk[i];
				y_tmp[i] = grad[i] - opt->grad[i];
			}
			save_lbfgs_pair(n, opt, s_tmp, y_tmp);
			free(s_tmp);
			free(y_tmp);
		}

		/* Save gradient for next (s,y) computation */
		memcpy(opt->grad, grad, n * sizeof(float));

		/* Print info */
		print_info_enriched(n, opt, fcost, OPT_NSTE);

		/* Test convergence */
		if (optim_test_conv(opt, fcost)) {
			*flag = OPT_CONV;
			print_info_enriched(n, opt, fcost, OPT_CONV);
			finalize_enriched(opt);
			return;
		}

		/* Handle indefinite Hessian: CG hit negative curvature on
		 * first iteration → force switch to L-BFGS */
		if (prev_method == 1 && opt->cpt_iter_CG == 0) {
			opt->enr_t = 1;
			opt->enr_force2 = 0;
			opt->enr_l = (3 * opt->enr_l / 2 < 30) ?
			              3 * opt->enr_l / 2 : 30;
			opt->enr_method = 0;
			opt->enr_k = 0;
		} else {
			/* Normal ADJUST */
			adjust_enriched(opt);
		}

		/* Eisenstat-Walker forcing term: only valid when previous
		 * step was HFN (residual from last CG is meaningful) */
		if (prev_method == 1 && opt->cpt_iter > 1) {
			forcing_term_enriched(n, grad, opt);
		}

		/* Prepare next phase (but don't start linesearch yet —
		 * the caller processes OPT_NSTE first, then calls us again) */
		if (opt->enr_method == 0) {
			/* L-BFGS mode: compute descent = -H(m)*grad.
			 * first_ls is already 1 from linesearch acceptance.
			 * Next call will enter linesearch section with first_ls. */
			optim_lbfgs_apply(n, opt, grad, opt->descent);
		} else {
			/* HFN mode: CG will start on next call */
			opt->enr_comm = TRN_DESC;
			opt->CG_phase = CG_INIT;
			opt->conv_CG = 0;
		}

		*flag = OPT_NSTE;

	} else if (opt->ls_task == LS_NEW_GRAD) {
		/* Linesearch needs another gradient */
		*flag = OPT_GRAD;
		opt->nfwd_pb++;

	} else if (opt->ls_task == LS_FAILURE) {
		/* Linesearch failed */
		if (opt->enr_method == 1) {
			/* HFN linesearch failed: restore x, request gradient
			 * recomputation, then switch to L-BFGS on next call */
			opt->enr_method = 0;
			opt->enr_k = 0;
			opt->enr_t = 1;
			opt->enr_l = (3 * opt->enr_l / 2 < 30) ?
			              3 * opt->enr_l / 2 : 30;

			/* Restore x to pre-step point */
			memcpy(x, opt->xk, n * sizeof(float));

			/* Compute L-BFGS descent from saved gradient */
			optim_lbfgs_apply(n, opt, opt->grad, opt->descent);

			/* Reset linesearch for fresh start */
			opt->first_ls = 1;

			/* Request gradient at restored x (caller recomputes) */
			*flag = OPT_GRAD;
			opt->nfwd_pb++;
		} else {
			/* L-BFGS linesearch failed: truly done */
			*flag = OPT_FAIL;
			memcpy(opt->grad, grad, n * sizeof(float));
			print_info_enriched(n, opt, fcost, OPT_FAIL);
			finalize_enriched(opt);
		}
	}
}
