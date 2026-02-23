/*
 * plbfgs.c - Preconditioned L-BFGS optimization algorithm.
 *
 * C translation of SEISCOPE Optimization Toolbox PLBFGS routines:
 *   PLBFGS.f90, init_PLBFGS.f90, descent_PLBFGS.f90, finalize_PLBFGS.f90
 *
 * The two-loop recursion is split into d escent1 (first backward loop)
 * and descent2 (forward loop) to allow the user to apply a preconditioner
 * to q_plb between the two loops.
 *
 * Reference: Nocedal & Wright, "Numerical Optimization", 2006
 *            Algorithm 7.4 p.178, Algorithm 7.5 p.179
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "optim.h"


/*--------------------------------------------------------------------
 * init_plbfgs -- Allocate arrays and initialize PLBFGS state.
 *
 * Translated from: PLBFGS/kernel/src/init_PLBFGS.f90
 *--------------------------------------------------------------------*/
static void init_plbfgs(int n, float *x, float fcost, float *grad,
                         float *grad_preco, optim_type *opt)
{
	int i;

	opt->n = n;

	/* Counters */
	opt->cpt_iter = 0;
	opt->f0 = fcost;
	opt->nfwd_pb = 0;
	opt->cpt_lbfgs = 1;

	/* Allocate L-BFGS history arrays */
	opt->sk = (float *)calloc((size_t)n * opt->l, sizeof(float));
	opt->yk = (float *)calloc((size_t)n * opt->l, sizeof(float));

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
	opt->q_plb = (float *)malloc(n * sizeof(float));

	/* First descent direction: d = -grad_preco */
	for (i = 0; i < n; i++)
		opt->descent[i] = -grad_preco[i];

	/* Save initial x, grad into history */
	optim_save_lbfgs(n, opt, x, grad);
}


/*--------------------------------------------------------------------
 * descent1_plbfgs -- First loop of two-loop recursion (backward).
 *
 * Computes q_plb and stores alpha_plb, rho_plb for the second loop.
 * After this, the user applies preconditioner to q_plb.
 *
 * Translated from: PLBFGS/kernel/src/descent_PLBFGS.f90 (descent1)
 *--------------------------------------------------------------------*/
static void descent1_plbfgs(int n, float *grad, optim_type *opt)
{
	int i, idx, borne;

	borne = opt->cpt_lbfgs - 1;

	/* Allocate per-iteration work arrays */
	opt->alpha_plb = (float *)malloc(borne * sizeof(float));
	opt->rho_plb = (float *)malloc(borne * sizeof(float));

	/* q = grad */
	memcpy(opt->q_plb, grad, n * sizeof(float));

	/* First loop: backward through history */
	for (i = 0; i < borne; i++) {
		idx = borne - 1 - i;

		opt->rho_plb[idx] = 1.0f / optim_dot(n, &opt->yk[idx * n],
		                                       &opt->sk[idx * n]);
		opt->alpha_plb[idx] = opt->rho_plb[idx] *
		                       optim_dot(n, &opt->sk[idx * n], opt->q_plb);

		/* q = q - alpha[idx] * yk[:,idx] */
		for (int j = 0; j < n; j++)
			opt->q_plb[j] -= opt->alpha_plb[idx] * opt->yk[idx * n + j];
	}
	/* Now user applies preconditioner to q_plb */
}


/*--------------------------------------------------------------------
 * descent2_plbfgs -- Second loop of two-loop recursion (forward).
 *
 * Called after user has applied preconditioner to q_plb.
 *
 * Translated from: PLBFGS/kernel/src/descent_PLBFGS.f90 (descent2)
 *--------------------------------------------------------------------*/
static void descent2_plbfgs(int n, float *grad, optim_type *opt)
{
	int i, idx, borne;
	float gamma_num, gamma_den, gamma, beta;

	borne = opt->cpt_lbfgs - 1;

	/* Scale by gamma */
	gamma_num = optim_dot(n, &opt->sk[(borne - 1) * n],
	                       &opt->yk[(borne - 1) * n]);
	gamma_den = optim_norm_l2(n, &opt->yk[(borne - 1) * n]);
	gamma = gamma_num / (gamma_den * gamma_den);

	for (i = 0; i < n; i++)
		opt->descent[i] = gamma * opt->q_plb[i];

	/* Second loop: forward through history */
	for (idx = 0; idx < borne; idx++) {
		beta = opt->rho_plb[idx] *
		       optim_dot(n, &opt->yk[idx * n], opt->descent);
		for (i = 0; i < n; i++)
			opt->descent[i] += (opt->alpha_plb[idx] - beta) *
			                    opt->sk[idx * n + i];
	}

	/* Negate for descent direction */
	for (i = 0; i < n; i++)
		opt->descent[i] = -opt->descent[i];

	/* Free per-iteration work arrays */
	free(opt->alpha_plb); opt->alpha_plb = NULL;
	free(opt->rho_plb); opt->rho_plb = NULL;
}


/*--------------------------------------------------------------------
 * finalize_plbfgs -- Free PLBFGS arrays.
 *
 * Translated from: PLBFGS/kernel/src/finalize_PLBFGS.f90
 *--------------------------------------------------------------------*/
static void finalize_plbfgs(optim_type *opt)
{
	if (opt->xk) { free(opt->xk); opt->xk = NULL; }
	if (opt->grad) { free(opt->grad); opt->grad = NULL; }
	if (opt->descent) { free(opt->descent); opt->descent = NULL; }
	if (opt->sk) { free(opt->sk); opt->sk = NULL; }
	if (opt->yk) { free(opt->yk); opt->yk = NULL; }
	if (opt->q_plb) { free(opt->q_plb); opt->q_plb = NULL; }
	if (opt->alpha_plb) { free(opt->alpha_plb); opt->alpha_plb = NULL; }
	if (opt->rho_plb) { free(opt->rho_plb); opt->rho_plb = NULL; }
}


/*--------------------------------------------------------------------
 * plbfgs_run -- Main PLBFGS reverse communication dispatcher.
 *
 * Like lbfgs_run but returns OPT_PREC when the user must apply
 * a preconditioner to opt->q_plb. After preconditioning, call
 * plbfgs_run again with the same flag.
 *
 * grad_preco: preconditioned gradient (used only at first iteration
 *             for the initial descent direction).
 *
 * Translated from: PLBFGS/kernel/src/PLBFGS.f90
 *--------------------------------------------------------------------*/
void plbfgs_run(int n, float *x, float fcost, float *grad,
                float *grad_preco, optim_type *opt, optFlag *flag)
{
	if (*flag == OPT_INIT) {
		/*--- Initialization ---*/
		init_plbfgs(n, x, fcost, grad, grad_preco, opt);
		optim_wolfe_linesearch(n, x, fcost, grad, opt);
		optim_print_info(n, "PL", opt, fcost, OPT_INIT);
		*flag = OPT_GRAD;
		opt->nfwd_pb++;

	} else if (*flag == OPT_PREC) {
		/*--- Return from user preconditioner: finish descent ---*/
		descent2_plbfgs(n, grad, opt);

		/* Save history */
		optim_save_lbfgs(n, opt, x, grad);
		opt->cpt_iter++;

		if (optim_test_conv(opt, fcost)) {
			*flag = OPT_CONV;
			memcpy(opt->grad, grad, n * sizeof(float));
			optim_print_info(n, "PL", opt, fcost, OPT_CONV);
			finalize_plbfgs(opt);
		} else {
			*flag = OPT_NSTE;
			memcpy(opt->grad, grad, n * sizeof(float));
			optim_print_info(n, "PL", opt, fcost, OPT_NSTE);
		}

	} else {
		/*--- Subsequent calls ---*/
		optim_wolfe_linesearch(n, x, fcost, grad, opt);

		if (opt->ls_task == LS_NEW_STEP) {
			/* Update history */
			optim_update_lbfgs(n, opt, x, grad);

			/* Start first loop of descent computation */
			descent1_plbfgs(n, grad, opt);

			/* Ask user to precondition q_plb */
			*flag = OPT_PREC;

		} else if (opt->ls_task == LS_NEW_GRAD) {
			*flag = OPT_GRAD;
			opt->nfwd_pb++;

		} else if (opt->ls_task == LS_FAILURE) {
			*flag = OPT_FAIL;
			memcpy(opt->grad, grad, n * sizeof(float));
			optim_print_info(n, "PL", opt, fcost, OPT_FAIL);
			finalize_plbfgs(opt);
		}
	}
}
