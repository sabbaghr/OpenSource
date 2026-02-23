/*
 * steepest_descent.c - Steepest descent optimization algorithm.
 *
 * Simple gradient descent with Wolfe linesearch, useful as a
 * baseline/debugging tool. Same reverse communication interface
 * as L-BFGS.
 *
 * C translation of SEISCOPE PSTD algorithm (simplified, no preconditioning).
 */

#include <stdlib.h>
#include <string.h>
#include "optim.h"


/*--------------------------------------------------------------------
 * init_sd -- Allocate arrays and initialize steepest descent state.
 *--------------------------------------------------------------------*/
static void init_sd(int n, float *x, float fcost, float *grad,
                    optim_type *opt)
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
	opt->xk = (float *)malloc(n * sizeof(float));
	memcpy(opt->xk, x, n * sizeof(float));

	opt->grad = (float *)malloc(n * sizeof(float));
	memcpy(opt->grad, grad, n * sizeof(float));

	opt->descent = (float *)malloc(n * sizeof(float));

	/* Descent direction: d = -grad */
	for (i = 0; i < n; i++)
		opt->descent[i] = -grad[i];
}


/*--------------------------------------------------------------------
 * finalize_sd -- Free steepest descent arrays.
 *--------------------------------------------------------------------*/
static void finalize_sd(optim_type *opt)
{
	if (opt->xk) { free(opt->xk); opt->xk = NULL; }
	if (opt->grad) { free(opt->grad); opt->grad = NULL; }
	if (opt->descent) { free(opt->descent); opt->descent = NULL; }
}


/*--------------------------------------------------------------------
 * steepest_descent_run -- Steepest descent reverse communication.
 *
 * Same interface as lbfgs_run. Descent direction is simply -grad
 * at each new iteration (no history).
 *--------------------------------------------------------------------*/
void steepest_descent_run(int n, float *x, float fcost, float *grad,
                          optim_type *opt, optFlag *flag)
{
	int i;

	if (*flag == OPT_INIT) {
		init_sd(n, x, fcost, grad, opt);
		optim_wolfe_linesearch(n, x, fcost, grad, opt);
		optim_print_info(n, "ST", opt, fcost, OPT_INIT);
		*flag = OPT_GRAD;
		opt->nfwd_pb++;

	} else {
		optim_wolfe_linesearch(n, x, fcost, grad, opt);

		if (opt->ls_task == LS_NEW_STEP) {
			opt->cpt_iter++;

			if (optim_test_conv(opt, fcost)) {
				*flag = OPT_CONV;
				memcpy(opt->grad, grad, n * sizeof(float));
				optim_print_info(n, "ST", opt, fcost, OPT_CONV);
				finalize_sd(opt);
			} else {
				*flag = OPT_NSTE;

				/* New descent direction: d = -grad */
				for (i = 0; i < n; i++)
					opt->descent[i] = -grad[i];

				memcpy(opt->grad, grad, n * sizeof(float));
				optim_print_info(n, "ST", opt, fcost, OPT_NSTE);
			}

		} else if (opt->ls_task == LS_NEW_GRAD) {
			*flag = OPT_GRAD;
			opt->nfwd_pb++;

		} else if (opt->ls_task == LS_FAILURE) {
			*flag = OPT_FAIL;
			memcpy(opt->grad, grad, n * sizeof(float));
			optim_print_info(n, "ST", opt, fcost, OPT_FAIL);
			finalize_sd(opt);
		}
	}
}
