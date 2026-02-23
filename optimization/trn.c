/*
 * trn.c - Truncated Newton optimization algorithm.
 *
 * C translation of SEISCOPE Optimization Toolbox TRN routines:
 *   TRN.f90, init_TRN.f90, descent_TRN.f90, forcing_term_TRN.f90,
 *   print_info_TRN.f90, finalize_TRN.f90
 *
 * Uses matrix-free conjugate gradient to approximately solve
 * H_k d_k = -grad_k, with Eisenstat-Walker stopping criterion.
 * The user provides Hessian-vector products via OPT_HESS flag.
 *
 * Reference: L. Metivier, R. Brossier, J. Virieux, S. Operto,
 *            "Truncated Newton and full waveform inversion", 2013,
 *            SIAM J. Sci. Comput., Vol. 35, No. 2, pp. B401-B437
 *            Nocedal & Wright, Algorithm 5.2 p.112
 *            Eisenstat & Walker, SIAM J. Sci. Comput. 17(1), 16-32, 1994
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "optim.h"


/* File handles for TRN convergence history */
static FILE *_trn_fp = NULL;     /* iterate_TRN.dat */
static FILE *_trn_cg_fp = NULL;  /* iterate_TRN_CG.dat */


/*--------------------------------------------------------------------
 * init_trn -- Allocate arrays and initialize TRN state.
 *
 * Translated from: TRN/kernel/src/init_TRN.f90
 *--------------------------------------------------------------------*/
static void init_trn(int n, float *x, float fcost, float *grad,
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
	if (opt->alpha <= 0.0f) opt->alpha = 1.0f;  /* respect pre-set alpha */

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

	/* Norm of the first gradient */
	opt->norm_grad = optim_norm_l2(n, grad);
}


/*--------------------------------------------------------------------
 * forcing_term_trn -- Compute Eisenstat-Walker forcing term.
 *
 * eta = ||grad - residual|| / ||grad_prev||
 * With safeguard: if eta_prev^phi > 0.1, eta = max(eta, eta_prev^phi)
 * where phi = (1+sqrt(5))/2 (golden ratio).
 *
 * Translated from: TRN/kernel/src/forcing_term_TRN.f90
 *--------------------------------------------------------------------*/
static void forcing_term_trn(int n, float *grad, optim_type *opt)
{
	int i;
	float eta_save, eta_save_power, norm_eisenvect;

	eta_save = opt->eta;

	for (i = 0; i < n; i++)
		opt->eisenvect[i] = grad[i] - opt->residual[i];

	norm_eisenvect = optim_norm_l2(n, opt->eisenvect);
	opt->eta = norm_eisenvect / opt->norm_grad_m1;

	/* Safeguard */
	eta_save_power = powf(eta_save, (1.0f + sqrtf(5.0f)) / 2.0f);
	if (eta_save_power > 0.1f)
		opt->eta = fmaxf(opt->eta, eta_save_power);

	if (opt->eta > 1.0f)
		opt->eta = 0.9f;
}


/*--------------------------------------------------------------------
 * descent_trn -- CG iteration for computing Newton descent direction.
 *
 * Matrix-free CG solver for H*descent = -grad.
 * Uses reverse communication: returns when Hessian-vector product
 * is needed (user sets opt->Hd = H * opt->d).
 *
 * Translated from: TRN/kernel/src/descent_TRN.f90
 *--------------------------------------------------------------------*/
static void descent_trn(int n, float *grad, optim_type *opt,
                        optFlag *flag)
{
	int i;
	float dHd, norm_residual_prev, alpha_cg, beta;

	if (opt->CG_phase == CG_INIT) {
		/* Initialize CG */
		memcpy(opt->residual, grad, n * sizeof(float));
		for (i = 0; i < n; i++) {
			opt->d[i] = -opt->residual[i];
			opt->Hd[i] = 0.0f;
			opt->descent[i] = 0.0f;
		}
		opt->qk_CG = 0.0f;
		opt->hessian_term = 0.0f;
		opt->norm_residual = optim_norm_l2(n, opt->residual);
		opt->conv_CG = 0;
		opt->cpt_iter_CG = 0;

		print_info_trn(n, opt, 0.0f, *flag);
		opt->CG_phase = CG_IRUN;

	} else {
		/* One CG iteration */
		dHd = optim_dot(n, opt->d, opt->Hd);

		if (dHd < 0.0f) {
			/* Negative curvature detected */
			opt->conv_CG = 1;

			if (opt->cpt_iter_CG == 0) {
				/* First iteration: use steepest descent */
				memcpy(opt->descent, opt->d, n * sizeof(float));

				if (opt->debug) {
					float grad_term;
					float *mgrad = (float *)malloc(n * sizeof(float));
					for (i = 0; i < n; i++)
						mgrad[i] = -grad[i];
					opt->norm_residual = optim_norm_l2(n, opt->residual);
					alpha_cg = (opt->norm_residual * opt->norm_residual) / dHd;
					opt->qkm1_CG = opt->qk_CG;
					grad_term = optim_dot(n, opt->descent, mgrad);
					opt->hessian_term += alpha_cg * alpha_cg * dHd;
					opt->qk_CG = -grad_term + 0.5f * opt->hessian_term;
					free(mgrad);
				}
			}

		} else {
			/* Normal CG iteration */
			opt->norm_residual = optim_norm_l2(n, opt->residual);
			alpha_cg = (opt->norm_residual * opt->norm_residual) / dHd;

			/* Update descent direction */
			memcpy(opt->descent_prev, opt->descent, n * sizeof(float));
			for (i = 0; i < n; i++) {
				opt->descent[i] += alpha_cg * opt->d[i];
				opt->residual[i] += alpha_cg * opt->Hd[i];
			}

			/* Update CG direction */
			norm_residual_prev = opt->norm_residual;
			opt->norm_residual = optim_norm_l2(n, opt->residual);
			beta = (opt->norm_residual * opt->norm_residual) /
			       (norm_residual_prev * norm_residual_prev);
			for (i = 0; i < n; i++)
				opt->d[i] = -opt->residual[i] + beta * opt->d[i];

			opt->cpt_iter_CG++;

			/* Debug: compute quadratic model */
			if (opt->debug) {
				float grad_term, descent_scal_Hd;
				float *mgrad = (float *)malloc(n * sizeof(float));
				for (i = 0; i < n; i++)
					mgrad[i] = -grad[i];
				opt->qkm1_CG = opt->qk_CG;
				grad_term = optim_dot(n, opt->descent, mgrad);
				descent_scal_Hd = optim_dot(n, opt->descent_prev, opt->Hd);
				opt->hessian_term += alpha_cg * alpha_cg * dHd +
				                     2.0f * alpha_cg * descent_scal_Hd;
				opt->qk_CG = -grad_term + 0.5f * opt->hessian_term;
				free(mgrad);
			}

			/* Check Eisenstat-Walker stopping criterion */
			opt->conv_CG = (opt->norm_residual <= opt->eta * opt->norm_grad) ||
			               (opt->cpt_iter_CG >= opt->niter_max_CG);

			print_info_trn(n, opt, 0.0f, *flag);
		}
	}
}


/*--------------------------------------------------------------------
 * finalize_trn -- Free TRN arrays.
 *
 * Translated from: TRN/kernel/src/finalize_TRN.f90
 *--------------------------------------------------------------------*/
static void finalize_trn(optim_type *opt)
{
	if (opt->xk) { free(opt->xk); opt->xk = NULL; }
	if (opt->grad) { free(opt->grad); opt->grad = NULL; }
	if (opt->descent) { free(opt->descent); opt->descent = NULL; }
	if (opt->descent_prev) { free(opt->descent_prev); opt->descent_prev = NULL; }
	if (opt->residual) { free(opt->residual); opt->residual = NULL; }
	if (opt->d) { free(opt->d); opt->d = NULL; }
	if (opt->Hd) { free(opt->Hd); opt->Hd = NULL; }
	if (opt->eisenvect) { free(opt->eisenvect); opt->eisenvect = NULL; }
}


/*--------------------------------------------------------------------
 * print_info_trn -- Write TRN convergence history to files.
 *
 * iterate_TRN.dat: outer iteration history
 * iterate_TRN_CG.dat: inner CG iteration history
 *
 * Translated from: TRN/kernel/src/print_info_TRN.f90
 *--------------------------------------------------------------------*/
void print_info_trn(int n, optim_type *opt, float fcost, optFlag flag)
{
	if (!opt->print_flag) return;

	if (flag == OPT_INIT) {
		_trn_fp = fopen("iterate_TRN.dat", "w");
		if (_trn_fp) {
			fprintf(_trn_fp,
				"******************************************************************************************\n");
			fprintf(_trn_fp,
				"                                 TRUNCATED NEWTON ALGORITHM\n");
			fprintf(_trn_fp,
				"******************************************************************************************\n");
			fprintf(_trn_fp, "     Convergence criterion  : %10.2e\n", opt->conv);
			fprintf(_trn_fp, "     Niter_max              : %7d\n", opt->niter_max);
			fprintf(_trn_fp, "     Initial cost is        : %10.2e\n", opt->f0);
			fprintf(_trn_fp, "     Initial norm_grad is   : %10.2e\n", opt->norm_grad);
			fprintf(_trn_fp, "     Maximum CG iter        : %7d\n", opt->niter_max_CG);
			fprintf(_trn_fp,
				"******************************************************************************************\n");
			fprintf(_trn_fp,
				"   Niter        fk      ||gk||      fk/f0"
				"         alpha       nls   nit_CG"
				"         eta    ngrad   nhess\n");
			fprintf(_trn_fp,
				"%6d%12.2e%12.2e%12.2e%12.2e%8d%8d%12.2e%8d%8d\n",
				opt->cpt_iter, fcost, opt->norm_grad,
				(opt->f0 > 0.0f) ? fcost / opt->f0 : 0.0f,
				opt->alpha, opt->cpt_ls, opt->cpt_iter_CG,
				opt->eta, opt->nfwd_pb, opt->nhess);
			fflush(_trn_fp);
		}

		_trn_cg_fp = fopen("iterate_TRN_CG.dat", "w");
		if (_trn_cg_fp) {
			fprintf(_trn_cg_fp,
				"******************************************************************************************\n");
			fprintf(_trn_cg_fp,
				"                                 TRUNCATED NEWTON ALGORITHM\n");
			fprintf(_trn_cg_fp,
				"                                      INNER CG HISTORY\n");
			fprintf(_trn_cg_fp,
				"******************************************************************************************\n");
			fprintf(_trn_cg_fp, "     Convergence criterion  : %10.2e\n", opt->conv);
			fprintf(_trn_cg_fp, "     Niter_max              : %7d\n", opt->niter_max);
			fprintf(_trn_cg_fp, "     Initial cost is        : %10.2e\n", opt->f0);
			fprintf(_trn_cg_fp, "     Initial norm_grad is   : %10.2e\n", opt->norm_grad);
			fprintf(_trn_cg_fp, "     Maximum CG iter        : %7d\n", opt->niter_max_CG);
			fprintf(_trn_cg_fp,
				"******************************************************************************************\n");
			fflush(_trn_cg_fp);
		}

	} else if (flag == OPT_CONV) {
		if (_trn_fp) {
			fprintf(_trn_fp,
				"**********************************************************************\n");
			if (opt->cpt_iter >= opt->niter_max)
				fprintf(_trn_fp,
					"  STOP: MAXIMUM NUMBER OF ITERATION REACHED\n");
			else
				fprintf(_trn_fp,
					"  STOP: CONVERGENCE CRITERION SATISFIED\n");
			fprintf(_trn_fp,
				"**********************************************************************\n");
			fclose(_trn_fp);
			_trn_fp = NULL;
		}
		if (_trn_cg_fp) {
			fclose(_trn_cg_fp);
			_trn_cg_fp = NULL;
		}

	} else if (flag == OPT_FAIL) {
		if (_trn_fp) {
			fprintf(_trn_fp,
				"**********************************************************************\n");
			fprintf(_trn_fp,
				"  STOP: LINESEARCH FAILURE\n");
			fprintf(_trn_fp,
				"**********************************************************************\n");
			fclose(_trn_fp);
			_trn_fp = NULL;
		}
		if (_trn_cg_fp) {
			fclose(_trn_cg_fp);
			_trn_cg_fp = NULL;
		}

	} else if (opt->comm == TRN_DESC) {
		/* CG iteration info */
		if (_trn_cg_fp) {
			if (opt->CG_phase == CG_INIT || opt->cpt_iter_CG == 0) {
				fprintf(_trn_cg_fp,
					"-------------------------------------------------------------------------------------------------\n");
				fprintf(_trn_cg_fp,
					" NONLINEAR ITERATION %4d ETA IS : %12.2e\n",
					opt->cpt_iter, opt->eta);
				fprintf(_trn_cg_fp,
					"-------------------------------------------------------------------------------------------------\n");
				fprintf(_trn_cg_fp,
					"  Iter_CG           qk     norm_res   norm_res/||gk||\n");
				fprintf(_trn_cg_fp,
					"%8d%12.2e%12.2e%12.2e\n",
					opt->cpt_iter_CG, opt->qk_CG,
					opt->norm_residual,
					(opt->norm_grad > 0.0f) ?
					opt->norm_residual / opt->norm_grad : 0.0f);
			} else {
				fprintf(_trn_cg_fp,
					"%8d%12.2e%12.2e%12.2e\n",
					opt->cpt_iter_CG, opt->qk_CG,
					opt->norm_residual,
					(opt->norm_grad > 0.0f) ?
					opt->norm_residual / opt->norm_grad : 0.0f);
			}
			fflush(_trn_cg_fp);
		}

	} else {
		/* Outer iteration info (from linesearch) */
		if (_trn_fp) {
			fprintf(_trn_fp,
				"%6d%12.2e%12.2e%12.2e%12.2e%8d%8d%12.2e%8d%8d\n",
				opt->cpt_iter, fcost, opt->norm_grad,
				(opt->f0 > 0.0f) ? fcost / opt->f0 : 0.0f,
				opt->alpha, opt->cpt_ls, opt->cpt_iter_CG,
				opt->eta, opt->nfwd_pb, opt->nhess);
			fflush(_trn_fp);
		}
	}
}


/*--------------------------------------------------------------------
 * trn_run -- Main TRN reverse communication dispatcher.
 *
 * Two nested loops:
 *   Outer: linesearch for step size
 *   Inner: CG for descent direction (requires Hessian-vector products)
 *
 * Returns:
 *   OPT_GRAD: user must compute cost and gradient at current x
 *   OPT_HESS: user must compute opt->Hd = H * opt->d
 *   OPT_NSTE: new step accepted
 *   OPT_CONV: converged
 *   OPT_FAIL: linesearch failed
 *
 * Translated from: TRN/kernel/src/TRN.f90
 *--------------------------------------------------------------------*/
void trn_run(int n, float *x, float fcost, float *grad,
             optim_type *opt, optFlag *flag)
{
	if (*flag == OPT_INIT) {
		/*--- Initialization ---*/
		init_trn(n, x, fcost, grad, opt);
		print_info_trn(n, opt, fcost, OPT_INIT);
		opt->comm = TRN_DESC;
		opt->CG_phase = CG_INIT;
		opt->nfwd_pb++;
		opt->conv_CG = 0;
		*flag = OPT_INIT;  /* Will fall through to DESC below */
	}

	if (opt->comm == TRN_DESC) {
		/*--- Computing descent direction via CG ---*/
		descent_trn(n, grad, opt, flag);

		if (opt->conv_CG) {
			/* CG converged: proceed to linesearch */
			opt->comm = TRN_NSTE;
			opt->CG_phase = CG_INIT;

			/* Initialize linesearch with the descent direction */
			optim_wolfe_linesearch(n, x, fcost, grad, opt);
			*flag = OPT_GRAD;
			opt->nfwd_pb++;
		} else {
			/* Need Hessian-vector product */
			*flag = OPT_HESS;
			opt->nhess++;
		}

	} else if (opt->comm == TRN_NSTE) {
		/*--- Linesearch for new step ---*/
		optim_wolfe_linesearch(n, x, fcost, grad, opt);

		if (opt->ls_task == LS_NEW_STEP) {
			opt->cpt_iter++;

			/* Save previous gradient norm, compute new one */
			opt->norm_grad_m1 = opt->norm_grad;
			opt->norm_grad = optim_norm_l2(n, grad);

			/* Print outer iteration info */
			print_info_trn(n, opt, fcost, OPT_NSTE);

			if (optim_test_conv(opt, fcost)) {
				*flag = OPT_CONV;
				print_info_trn(n, opt, fcost, OPT_CONV);
				finalize_trn(opt);
			} else {
				*flag = OPT_NSTE;
				opt->comm = TRN_DESC;

				/* Update forcing term */
				forcing_term_trn(n, grad, opt);
			}

		} else if (opt->ls_task == LS_NEW_GRAD) {
			*flag = OPT_GRAD;
			opt->nfwd_pb++;

		} else if (opt->ls_task == LS_FAILURE) {
			*flag = OPT_FAIL;
			print_info_trn(n, opt, fcost, OPT_FAIL);
			finalize_trn(opt);
		}
	}
}
