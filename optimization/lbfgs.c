/*
 * lbfgs.c - L-BFGS optimization algorithm.
 *
 * C translation of SEISCOPE Optimization Toolbox L-BFGS routines:
 *   LBFGS.f90, init_LBFGS.f90, descent_LBFGS.f90,
 *   save_LBFGS.f90, update_LBFGS.f90, finalize_LBFGS.f90
 *
 * Reference: Nocedal & Wright, "Numerical Optimization", 2006
 *            Algorithm 7.4 p.178, Algorithm 7.5 p.179
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "optim.h"


/*--------------------------------------------------------------------
 * init_lbfgs -- Allocate arrays and initialize L-BFGS state.
 *
 * Sets counters, Wolfe parameters, allocates history arrays sk/yk,
 * computes initial descent direction d = -grad.
 *
 * Translated from: LBFGS/kernel/src/init_LBFGS.f90
 *--------------------------------------------------------------------*/
static void init_lbfgs(int n, float *x, float fcost, float *grad,
                       optim_type *opt)
{
	int i;

	/* Store dimension */
	opt->n = n;

	/* Counters */
	opt->cpt_iter = 0;
	opt->f0 = fcost;
	opt->nfwd_pb = 0;

	/* Allocate L-BFGS history arrays (flat n*l, column-major) */
	opt->sk = (float *)calloc((size_t)n * opt->l, sizeof(float));
	opt->yk = (float *)calloc((size_t)n * opt->l, sizeof(float));
	opt->cpt_lbfgs = 1;  /* 1-based counter (matches Fortran convention) */

	/* Linesearch parameters */
	opt->m1 = 1e-4f;
	opt->m2 = 0.9f;
	opt->mult_factor = 10.0f;
	opt->fk = fcost;
	if (opt->nls_max <= 0) opt->nls_max = 20;
	opt->cpt_ls = 0;
	opt->first_ls = 1;
	if (opt->alpha <= 0.0f) opt->alpha = 1.0f;  /* respect pre-set alpha */

	/* Work arrays */
	opt->xk = (float *)malloc(n * sizeof(float));
	memcpy(opt->xk, x, n * sizeof(float));

	opt->grad = (float *)malloc(n * sizeof(float));
	memcpy(opt->grad, grad, n * sizeof(float));

	opt->descent = (float *)malloc(n * sizeof(float));

	/* Initial descent direction: d = -grad */
	for (i = 0; i < n; i++)
		opt->descent[i] = -grad[i];

	/* Save initial x, grad into history slot 1 (for later differencing) */
	optim_save_lbfgs(n, opt, x, grad);
}


/*--------------------------------------------------------------------
 * update_lbfgs -- Convert saved x,grad to step/gradient differences.
 *
 * sk[:,j] = x_new - x_old  (stored in-place)
 * yk[:,j] = grad_new - grad_old  (stored in-place)
 *
 * If buffer not full, increments cpt_lbfgs.
 * If buffer full, updates only the last slot (index l-1 in C).
 *
 * Translated from: LBFGS/kernel/src/update_LBFGS.f90
 *--------------------------------------------------------------------*/
void optim_update_lbfgs(int n, optim_type *opt, float *x, float *grad)
{
	int i;
	int j;  /* column index (0-based) */

	if (opt->cpt_lbfgs <= opt->l) {
		/* Buffer not full: update slot cpt_lbfgs-1 (0-based) */
		j = opt->cpt_lbfgs - 1;
		for (i = 0; i < n; i++) {
			opt->sk[j * n + i] = x[i] - opt->sk[j * n + i];
			opt->yk[j * n + i] = grad[i] - opt->yk[j * n + i];
		}
		opt->cpt_lbfgs++;
	} else {
		/* Buffer full: update last slot (l-1 in 0-based) */
		j = opt->l - 1;
		for (i = 0; i < n; i++) {
			opt->sk[j * n + i] = x[i] - opt->sk[j * n + i];
			opt->yk[j * n + i] = grad[i] - opt->yk[j * n + i];
		}
	}
}


/*--------------------------------------------------------------------
 * save_lbfgs -- Save current x, grad into history buffer.
 *
 * If buffer not full: store at position cpt_lbfgs-1.
 * If buffer full: shift all columns left by 1, store at last position.
 *
 * Translated from: LBFGS/kernel/src/save_LBFGS.f90
 *--------------------------------------------------------------------*/
void optim_save_lbfgs(int n, optim_type *opt, float *x, float *grad)
{
	int i, j;

	if (opt->cpt_lbfgs <= opt->l) {
		/* Not full: store at cpt_lbfgs-1 (0-based) */
		j = opt->cpt_lbfgs - 1;
		memcpy(&opt->sk[j * n], x, n * sizeof(float));
		memcpy(&opt->yk[j * n], grad, n * sizeof(float));
	} else {
		/* Full: shift columns left, store new at end */
		for (j = 0; j < opt->l - 1; j++) {
			memcpy(&opt->sk[j * n], &opt->sk[(j + 1) * n],
			       n * sizeof(float));
			memcpy(&opt->yk[j * n], &opt->yk[(j + 1) * n],
			       n * sizeof(float));
		}
		j = opt->l - 1;
		memcpy(&opt->sk[j * n], x, n * sizeof(float));
		memcpy(&opt->yk[j * n], grad, n * sizeof(float));
	}
}


/*--------------------------------------------------------------------
 * descent_lbfgs -- Two-loop recursion for L-BFGS descent direction.
 *
 * Computes d = -H_k * grad where H_k is the L-BFGS approximation
 * of the inverse Hessian (Nocedal Algorithm 7.5, p.179).
 *
 * Safeguard: falls back to steepest descent if sk or yk norms are zero.
 *
 * Translated from: LBFGS/kernel/src/descent_LBFGS.f90
 *--------------------------------------------------------------------*/
static void descent_lbfgs(int n, optim_type *opt, float *grad)
{
	int i, j, idx;
	int borne;  /* effective number of stored pairs */
	float norm_sk, norm_yk;
	float gamma_num, gamma_den, gamma;
	float beta;
	float *q, *alpha, *rho;

	/* borne = cpt_lbfgs - 1 (number of completed pairs) */
	borne = opt->cpt_lbfgs - 1;
	if (borne <= 0) {
		/* No history: steepest descent */
		for (i = 0; i < n; i++)
			opt->descent[i] = -grad[i];
		return;
	}

	/* Safeguard: check norms of most recent pair */
	/* Most recent pair is at column borne-1 (0-based) */
	norm_sk = optim_norm_l2(n, &opt->sk[(borne - 1) * n]);
	norm_yk = optim_norm_l2(n, &opt->yk[(borne - 1) * n]);

	if (norm_sk == 0.0f || norm_yk == 0.0f) {
		for (i = 0; i < n; i++)
			opt->descent[i] = -grad[i];
		return;
	}

	/* Allocate work arrays */
	alpha = (float *)malloc(borne * sizeof(float));
	rho   = (float *)malloc(borne * sizeof(float));
	q     = (float *)malloc(n * sizeof(float));

	/* q = grad */
	memcpy(q, grad, n * sizeof(float));

	/*--- First loop: backward through history (newest to oldest) ---*/
	for (i = 0; i < borne; i++) {
		idx = borne - 1 - i;  /* reverse: borne-1, borne-2, ..., 0 */

		rho[idx] = 1.0f / optim_dot(n, &opt->yk[idx * n],
		                              &opt->sk[idx * n]);
		alpha[idx] = rho[idx] * optim_dot(n, &opt->sk[idx * n], q);

		/* q = q - alpha[idx] * yk[:,idx] */
		for (j = 0; j < n; j++)
			q[j] -= alpha[idx] * opt->yk[idx * n + j];
	}

	/*--- Scale by gamma (initial inverse Hessian approximation) ---*/
	gamma_num = optim_dot(n, &opt->sk[(borne - 1) * n],
	                       &opt->yk[(borne - 1) * n]);
	gamma_den = optim_norm_l2(n, &opt->yk[(borne - 1) * n]);
	gamma = gamma_num / (gamma_den * gamma_den);

	/* descent = gamma * q */
	for (i = 0; i < n; i++)
		opt->descent[i] = gamma * q[i];

	/*--- Second loop: forward through history (oldest to newest) ---*/
	for (idx = 0; idx < borne; idx++) {
		beta = rho[idx] * optim_dot(n, &opt->yk[idx * n],
		                             opt->descent);
		/* descent = descent + (alpha[idx] - beta) * sk[:,idx] */
		for (j = 0; j < n; j++)
			opt->descent[j] += (alpha[idx] - beta) * opt->sk[idx * n + j];
	}

	/*--- Negate for descent direction ---*/
	for (i = 0; i < n; i++)
		opt->descent[i] = -opt->descent[i];

	free(q);
	free(alpha);
	free(rho);
}


/*--------------------------------------------------------------------
 * optim_finalize -- Free all internally allocated arrays.
 *
 * Translated from: LBFGS/kernel/src/finalize_LBFGS.f90
 *--------------------------------------------------------------------*/
void optim_finalize(optim_type *opt)
{
	if (opt->sk) { free(opt->sk); opt->sk = NULL; }
	if (opt->yk) { free(opt->yk); opt->yk = NULL; }
	if (opt->xk) { free(opt->xk); opt->xk = NULL; }
	if (opt->grad) { free(opt->grad); opt->grad = NULL; }
	if (opt->descent) { free(opt->descent); opt->descent = NULL; }
	/* PLBFGS */
	if (opt->q_plb) { free(opt->q_plb); opt->q_plb = NULL; }
	if (opt->alpha_plb) { free(opt->alpha_plb); opt->alpha_plb = NULL; }
	if (opt->rho_plb) { free(opt->rho_plb); opt->rho_plb = NULL; }
	/* PNLCG */
	if (opt->grad_prev) { free(opt->grad_prev); opt->grad_prev = NULL; }
	if (opt->descent_prev) { free(opt->descent_prev); opt->descent_prev = NULL; }
	/* TRN */
	if (opt->residual) { free(opt->residual); opt->residual = NULL; }
	if (opt->d) { free(opt->d); opt->d = NULL; }
	if (opt->Hd) { free(opt->Hd); opt->Hd = NULL; }
	if (opt->eisenvect) { free(opt->eisenvect); opt->eisenvect = NULL; }
}


/*--------------------------------------------------------------------
 * lbfgs_run -- Main L-BFGS reverse communication dispatcher.
 *
 * Call with flag=OPT_INIT on first call (x, fcost, grad must be set).
 * On return, check flag:
 *   OPT_GRAD: compute cost and gradient at updated x, call again.
 *   OPT_NSTE: new iteration completed, call again.
 *   OPT_CONV: converged.
 *   OPT_FAIL: linesearch failed.
 *
 * Translated from: LBFGS/kernel/src/LBFGS.f90
 *--------------------------------------------------------------------*/
void lbfgs_run(int n, float *x, float fcost, float *grad,
               optim_type *opt, optFlag *flag)
{
	int i;

	if (*flag == OPT_INIT) {
		/*--- Initialization ---*/
		init_lbfgs(n, x, fcost, grad, opt);
		optim_wolfe_linesearch(n, x, fcost, grad, opt);
		optim_print_info(n, "LB", opt, fcost, OPT_INIT);
		*flag = OPT_GRAD;
		opt->nfwd_pb++;

	} else {
		/*--- Subsequent calls ---*/
		optim_wolfe_linesearch(n, x, fcost, grad, opt);

		if (opt->ls_task == LS_NEW_STEP) {
			/* Step accepted: test convergence */
			opt->cpt_iter++;

			if (optim_test_conv(opt, fcost)) {
				/* Converged */
				*flag = OPT_CONV;
				memcpy(opt->grad, grad, n * sizeof(float));
				optim_print_info(n, "LB", opt, fcost, OPT_CONV);
				optim_finalize(opt);
			} else {
				/* Continue: update history, compute new descent */
				*flag = OPT_NSTE;

				optim_update_lbfgs(n, opt, x, grad);
				descent_lbfgs(n, opt, grad);
				optim_save_lbfgs(n, opt, x, grad);

				memcpy(opt->grad, grad, n * sizeof(float));
				optim_print_info(n, "LB", opt, fcost, OPT_NSTE);
			}

		} else if (opt->ls_task == LS_NEW_GRAD) {
			/* Linesearch needs another gradient evaluation */
			*flag = OPT_GRAD;
			opt->nfwd_pb++;

		} else if (opt->ls_task == LS_FAILURE) {
			/* Linesearch failed */
			*flag = OPT_FAIL;
			memcpy(opt->grad, grad, n * sizeof(float));
			optim_print_info(n, "LB", opt, fcost, OPT_FAIL);
			optim_finalize(opt);
		}
	}
}
