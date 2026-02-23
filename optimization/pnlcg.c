/*
 * pnlcg.c - Preconditioned nonlinear conjugate gradient (Dai-Yuan).
 *
 * C translation of SEISCOPE Optimization Toolbox PNLCG routines:
 *   PNLCG.f90, init_PNLCG.f90, descent_PNLCG.f90, finalize_PNLCG.f90
 *
 * Reference: Y. Dai and Y. Yuan, "A nonlinear conjugate gradient method
 *            with a strong global convergence property", SIAM J. Optim.,
 *            Vol. 10, pp. 177-182, 1999.
 *            Nocedal & Wright, "Numerical Optimization", 2006, p.132
 */

#include <stdlib.h>
#include <string.h>
#include "optim.h"


/*--------------------------------------------------------------------
 * init_pnlcg -- Allocate arrays and initialize PNLCG state.
 *
 * Translated from: PNLCG/kernel/src/init_PNLCG.f90
 *--------------------------------------------------------------------*/
static void init_pnlcg(int n, float *x, float fcost, float *grad,
                        float *grad_preco, optim_type *opt)
{
	int i;

	opt->n = n;

	/* Counters */
	opt->cpt_iter = 0;
	opt->f0 = fcost;
	opt->nfwd_pb = 0;

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
	opt->grad_prev = (float *)malloc(n * sizeof(float));
	opt->descent_prev = (float *)malloc(n * sizeof(float));

	opt->xk = (float *)malloc(n * sizeof(float));
	memcpy(opt->xk, x, n * sizeof(float));

	opt->grad = (float *)malloc(n * sizeof(float));
	memcpy(opt->grad, grad, n * sizeof(float));

	opt->descent = (float *)malloc(n * sizeof(float));

	/* First descent direction: d = -grad_preco */
	for (i = 0; i < n; i++)
		opt->descent[i] = -grad_preco[i];
}


/*--------------------------------------------------------------------
 * descent_pnlcg -- Compute Dai-Yuan CG descent direction.
 *
 * beta = <grad, grad_preco> / <(grad - grad_prev), descent_prev>
 * d = -grad_preco + beta * descent_prev
 *
 * Safeguard: if |beta| >= 1e5, reset to steepest descent (beta=0).
 *
 * Translated from: PNLCG/kernel/src/descent_PNLCG.f90
 *--------------------------------------------------------------------*/
static void descent_pnlcg(int n, float *grad, float *grad_preco,
                           optim_type *opt)
{
	int i;
	float gkpgk, skpk, beta;
	float *sk;

	/* Store old descent direction */
	memcpy(opt->descent_prev, opt->descent, n * sizeof(float));

	/* Compute beta (Dai-Yuan formula) */
	gkpgk = optim_dot(n, grad, grad_preco);

	sk = (float *)malloc(n * sizeof(float));
	for (i = 0; i < n; i++)
		sk[i] = grad[i] - opt->grad_prev[i];

	skpk = optim_dot(n, sk, opt->descent_prev);
	beta = gkpgk / skpk;

	/* Safeguard */
	if (beta >= 1e5f || beta <= -1e5f)
		beta = 0.0f;

	/* New descent direction */
	for (i = 0; i < n; i++)
		opt->descent[i] = -grad_preco[i] + beta * opt->descent_prev[i];

	free(sk);
}


/*--------------------------------------------------------------------
 * finalize_pnlcg -- Free PNLCG arrays.
 *
 * Translated from: PNLCG/kernel/src/finalize_PNLCG.f90
 *--------------------------------------------------------------------*/
static void finalize_pnlcg(optim_type *opt)
{
	if (opt->xk) { free(opt->xk); opt->xk = NULL; }
	if (opt->grad) { free(opt->grad); opt->grad = NULL; }
	if (opt->grad_prev) { free(opt->grad_prev); opt->grad_prev = NULL; }
	if (opt->descent) { free(opt->descent); opt->descent = NULL; }
	if (opt->descent_prev) { free(opt->descent_prev); opt->descent_prev = NULL; }
}


/*--------------------------------------------------------------------
 * pnlcg_run -- PNLCG reverse communication dispatcher.
 *
 * Same interface as lbfgs_run but takes an additional grad_preco
 * parameter (preconditioned gradient). If no preconditioning is
 * desired, pass grad_preco = grad.
 *
 * Translated from: PNLCG/kernel/src/PNLCG.f90
 *--------------------------------------------------------------------*/
void pnlcg_run(int n, float *x, float fcost, float *grad,
               float *grad_preco, optim_type *opt, optFlag *flag)
{
	if (*flag == OPT_INIT) {
		/*--- Initialization ---*/
		init_pnlcg(n, x, fcost, grad, grad_preco, opt);
		optim_wolfe_linesearch(n, x, fcost, grad, opt);
		optim_print_info(n, "CG", opt, fcost, OPT_INIT);
		*flag = OPT_GRAD;
		opt->nfwd_pb++;
		/* Store current gradient before user computes the new one */
		memcpy(opt->grad_prev, grad, n * sizeof(float));

	} else {
		optim_wolfe_linesearch(n, x, fcost, grad, opt);

		if (opt->ls_task == LS_NEW_STEP) {
			opt->cpt_iter++;

			if (optim_test_conv(opt, fcost)) {
				*flag = OPT_CONV;
				memcpy(opt->grad, grad, n * sizeof(float));
				optim_print_info(n, "CG", opt, fcost, OPT_CONV);
				finalize_pnlcg(opt);
			} else {
				*flag = OPT_NSTE;
				descent_pnlcg(n, grad, grad_preco, opt);
				memcpy(opt->grad, grad, n * sizeof(float));
				optim_print_info(n, "CG", opt, fcost, OPT_NSTE);
			}

		} else if (opt->ls_task == LS_NEW_GRAD) {
			*flag = OPT_GRAD;
			opt->nfwd_pb++;
			/* Store current gradient before user computes the new one */
			memcpy(opt->grad_prev, grad, n * sizeof(float));

		} else if (opt->ls_task == LS_FAILURE) {
			*flag = OPT_FAIL;
			memcpy(opt->grad, grad, n * sizeof(float));
			optim_print_info(n, "CG", opt, fcost, OPT_FAIL);
			finalize_pnlcg(opt);
		}
	}
}
