/*
 * optim_common.c - Shared optimization routines.
 *
 * C translation of SEISCOPE Optimization Toolbox common utilities:
 *   normL2.f90, scalL2.f90, project.f90, std_test_conv.f90,
 *   std_linesearch.f90, print_info.f90
 *
 * Reference: SEISCOPE Optimization Toolbox (Métivier & Brossier, 2016)
 *            Nocedal & Wright, "Numerical Optimization", 2006
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "optim.h"


/*--------------------------------------------------------------------
 * optim_norm_l2 -- Euclidean norm ||x||_2
 *
 * Translated from: COMMON/src/normL2.f90
 *--------------------------------------------------------------------*/
float optim_norm_l2(int n, const float *x)
{
	double sum = 0.0;
	int i;
	for (i = 0; i < n; i++)
		sum += (double)x[i] * (double)x[i];
	return (float)sqrt(sum);
}


/*--------------------------------------------------------------------
 * optim_dot -- Dot product <x, y>
 *
 * Translated from: COMMON/src/scalL2.f90
 *--------------------------------------------------------------------*/
float optim_dot(int n, const float *x, const float *y)
{
	double sum = 0.0;
	int i;
	for (i = 0; i < n; i++)
		sum += (double)x[i] * (double)y[i];
	return (float)sum;
}


/*--------------------------------------------------------------------
 * optim_project -- Box constraint projection.
 *
 * Clamps each element of x to [lb+threshold, ub-threshold].
 *
 * Translated from: COMMON/src/project.f90
 *--------------------------------------------------------------------*/
void optim_project(int n, optim_type *opt, float *x)
{
	int i;
	if (!opt->bound || !opt->lb || !opt->ub) return;
	for (i = 0; i < n; i++) {
		if (x[i] > opt->ub[i])
			x[i] = opt->ub[i] - opt->threshold;
		if (x[i] < opt->lb[i])
			x[i] = opt->lb[i] + opt->threshold;
	}
}


/*--------------------------------------------------------------------
 * optim_test_conv -- Convergence test.
 *
 * Returns 1 if: fcost/f0 < conv  OR  cpt_iter >= niter_max
 *
 * Translated from: COMMON/src/std_test_conv.f90
 *--------------------------------------------------------------------*/
int optim_test_conv(optim_type *opt, float fcost)
{
	if (opt->f0 > 0.0f && (fcost / opt->f0) < opt->conv)
		return 1;
	if (opt->cpt_iter >= opt->niter_max)
		return 1;
	return 0;
}


/*--------------------------------------------------------------------
 * optim_print_info -- Write convergence history to file.
 *
 * tag: "ST" (steepest descent), "LB" (L-BFGS)
 * flag: OPT_INIT (write header), OPT_CONV/OPT_FAIL (close file),
 *       OPT_NSTE or other (write iteration row)
 *
 * Output file: iterate_XX.dat where XX = tag
 *
 * Translated from: COMMON/src/print_info.f90
 *--------------------------------------------------------------------*/

/* File handle persists across calls (like Fortran unit 10) */
static FILE *_print_fp = NULL;

void optim_print_info(int n, const char *tag, optim_type *opt,
                      float fcost, optFlag flag)
{
	float ng;

	if (!opt->print_flag) return;

	ng = optim_norm_l2(n, opt->grad);

	if (flag == OPT_INIT) {
		char fname[64];
		snprintf(fname, sizeof(fname), "iterate_%s.dat", tag);
		_print_fp = fopen(fname, "w");
		if (!_print_fp) return;

		fprintf(_print_fp,
			"**********************************************************************\n");
		if (strcmp(tag, "ST") == 0)
			fprintf(_print_fp,
				"         STEEPEST DESCENT ALGORITHM\n");
		else if (strcmp(tag, "LB") == 0)
			fprintf(_print_fp,
				"             l-BFGS ALGORITHM\n");
		fprintf(_print_fp,
			"**********************************************************************\n");
		fprintf(_print_fp, "     Convergence criterion  : %10.2e\n", opt->conv);
		fprintf(_print_fp, "     Niter_max              : %7d\n", opt->niter_max);
		fprintf(_print_fp, "     Initial cost is        : %10.2e\n", opt->f0);
		fprintf(_print_fp, "     Initial norm_grad is   : %10.2e\n", ng);
		fprintf(_print_fp,
			"**********************************************************************\n");
		fprintf(_print_fp,
			"   Niter        fk      ||gk||      fk/f0"
			"         alpha       nls     ngrad\n");
		fprintf(_print_fp,
			"%6d%12.2e%12.2e%12.2e%12.2e%8d%8d\n",
			opt->cpt_iter, fcost, ng,
			(opt->f0 > 0.0f) ? fcost / opt->f0 : 0.0f,
			opt->alpha, opt->cpt_ls, opt->nfwd_pb);
		fflush(_print_fp);

	} else if (flag == OPT_CONV) {
		if (!_print_fp) return;
		fprintf(_print_fp,
			"%6d%12.2e%12.2e%12.2e%12.2e%8d%8d\n",
			opt->cpt_iter, fcost, ng,
			(opt->f0 > 0.0f) ? fcost / opt->f0 : 0.0f,
			opt->alpha, opt->cpt_ls, opt->nfwd_pb);
		fprintf(_print_fp,
			"**********************************************************************\n");
		if (opt->cpt_iter >= opt->niter_max)
			fprintf(_print_fp,
				"  STOP: MAXIMUM NUMBER OF ITERATION REACHED\n");
		else
			fprintf(_print_fp,
				"  STOP: CONVERGENCE CRITERION SATISFIED\n");
		fprintf(_print_fp,
			"**********************************************************************\n");
		fclose(_print_fp);
		_print_fp = NULL;

	} else if (flag == OPT_FAIL) {
		if (!_print_fp) return;
		fprintf(_print_fp,
			"**********************************************************************\n");
		fprintf(_print_fp,
			"  STOP: LINESEARCH FAILURE\n");
		fprintf(_print_fp,
			"**********************************************************************\n");
		fclose(_print_fp);
		_print_fp = NULL;

	} else {
		/* OPT_NSTE or any other: write iteration row */
		if (!_print_fp) return;
		fprintf(_print_fp,
			"%6d%12.2e%12.2e%12.2e%12.2e%8d%8d\n",
			opt->cpt_iter, fcost, ng,
			(opt->f0 > 0.0f) ? fcost / opt->f0 : 0.0f,
			opt->alpha, opt->cpt_ls, opt->nfwd_pb);
		fflush(_print_fp);
	}
}


/*--------------------------------------------------------------------
 * optim_wolfe_linesearch -- Wolfe condition linesearch.
 *
 * Implements bracketing + bisection strategy enforcing:
 *   Condition 1 (sufficient decrease): f <= fk + m1*alpha*q0
 *   Condition 2 (curvature):           q >= m2*q0
 * where q0 = <grad_initial, descent>, q = <grad_current, descent>
 *
 * Updates opt->ls_task:
 *   LS_NEW_GRAD  -- need new cost/gradient at updated x
 *   LS_NEW_STEP  -- step accepted
 *   LS_FAILURE   -- linesearch failed
 *
 * Translated from: COMMON/src/std_linesearch.f90
 *--------------------------------------------------------------------*/
void optim_wolfe_linesearch(int n, float *x, float fcost, float *grad,
                            optim_type *opt)
{
	int i;
	float new_alpha;
	float armijo_bound;  /* fk + m1*alpha*q0 */

	if (opt->first_ls) {
		/*----------------------------------------------
		 * FIRST LINESEARCH CALL: initialization step
		 *----------------------------------------------*/
		opt->fk = fcost;
		opt->q0 = optim_dot(n, grad, opt->descent);

		/* Initialize search interval bounds to 0 */
		opt->alpha_L = 0.0f;
		opt->alpha_R = 0.0f;

		opt->ls_task = LS_NEW_GRAD;
		opt->first_ls = 0;

		/* Save current x */
		for (i = 0; i < n; i++)
			opt->xk[i] = x[i];

		/* Take step: x = xk + alpha * descent */
		for (i = 0; i < n; i++)
			x[i] = opt->xk[i] + opt->alpha * opt->descent[i];

		if (opt->bound)
			optim_project(n, opt, x);

		opt->cpt_ls = 0;

	} else if (opt->cpt_ls >= opt->nls_max && fcost < opt->fk) {
		/*----------------------------------------------
		 * Max LS iterations reached but cost decreased:
		 * forced accept
		 *----------------------------------------------*/
		opt->ls_task = LS_NEW_STEP;
		opt->first_ls = 1;

		for (i = 0; i < n; i++)
			x[i] = opt->xk[i] + opt->alpha * opt->descent[i];

		if (opt->bound)
			optim_project(n, opt, x);

	} else if (opt->cpt_ls >= opt->nls_max) {
		/*----------------------------------------------
		 * Max LS iterations without decrease: failure
		 *----------------------------------------------*/
		opt->ls_task = LS_FAILURE;

	} else {
		/*----------------------------------------------
		 * Normal linesearch iteration
		 *----------------------------------------------*/
		opt->q = optim_dot(n, grad, opt->descent);
		armijo_bound = opt->fk + opt->m1 * opt->alpha * opt->q0;

		if (fcost <= armijo_bound && opt->q >= opt->m2 * opt->q0) {
			/*--- Both Wolfe conditions satisfied: accept ---*/
			opt->ls_task = LS_NEW_STEP;
			opt->first_ls = 1;

			if (opt->debug && opt->print_flag && _print_fp) {
				fprintf(_print_fp, "  LS accept: fcost=%e fk=%e alpha=%e q=%e q0=%e cpt_ls=%d\n",
					fcost, opt->fk, opt->alpha, opt->q, opt->q0, opt->cpt_ls);
			}

		} else if (fcost > armijo_bound) {
			/*--- Sufficient decrease violated: shrink interval ---*/
			if (opt->debug && opt->print_flag && _print_fp) {
				fprintf(_print_fp, "  LS failure 1: fcost=%e fk=%e alpha=%e q0=%e cpt_ls=%d\n",
					fcost, opt->fk, opt->alpha, opt->q0, opt->cpt_ls);
			}
			opt->alpha_R = opt->alpha;
			new_alpha = (opt->alpha_L + opt->alpha_R) / 2.0f;
			opt->alpha = new_alpha;
			opt->ls_task = LS_NEW_GRAD;
			opt->cpt_ls++;

		} else {
			/*--- Curvature condition violated: expand or bisect ---*/
			if (opt->debug && opt->print_flag && _print_fp) {
				fprintf(_print_fp, "  LS failure 2: fcost=%e fk=%e alpha=%e q=%e q0=%e cpt_ls=%d\n",
					fcost, opt->fk, opt->alpha, opt->q, opt->q0, opt->cpt_ls);
			}
			opt->alpha_L = opt->alpha;
			if (opt->alpha_R != 0.0f)
				new_alpha = (opt->alpha_L + opt->alpha_R) / 2.0f;
			else
				new_alpha = opt->mult_factor * opt->alpha;
			opt->alpha = new_alpha;
			opt->ls_task = LS_NEW_GRAD;
			opt->cpt_ls++;
		}

		/* Update x for next evaluation */
		for (i = 0; i < n; i++)
			x[i] = opt->xk[i] + opt->alpha * opt->descent[i];

		if (opt->bound)
			optim_project(n, opt, x);
	}
}
