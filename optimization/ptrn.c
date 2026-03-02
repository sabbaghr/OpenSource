/*
 * ptrn.c - Preconditioned Truncated Newton optimization algorithm.
 *
 * C translation of SEISCOPE Optimization Toolbox PTRN routines:
 *   PTRN.f90, init_PTRN.f90, descent_PTRN.f90, finalize_PTRN.f90,
 *   print_info_PTRN.f90
 *
 * Like TRN but uses preconditioned CG (Nocedal Algorithm 5.3 p.119):
 *   Standard CG:       alpha = <r,r>/<d,Hd>,   beta = <r_new,r_new>/<r_old,r_old>
 *   Preconditioned CG: alpha = <r,z>/<d,Hd>,   beta = <r_new,z_new>/<r_old,z_old>
 * where z = P^{-1} r (preconditioned residual).
 *
 * The CG iteration is split into three phases for reverse communication:
 *   PTRN0: initialization (r=grad, z=grad_preco, d=-z)
 *   PTRN1: compute alpha, update descent/residual -> return OPT_PREC
 *   (user sets residual_preco = P^{-1} * residual)
 *   PTRN2: compute beta, update CG direction d -> return OPT_HESS
 *
 * Reference: L. Metivier, R. Brossier, J. Virieux, S. Operto,
 *            "Truncated Newton and full waveform inversion", 2013,
 *            SIAM J. Sci. Comput., Vol. 35, No. 2, pp. B401-B437
 *            Nocedal & Wright, Algorithm 5.3 p.119
 *            Eisenstat & Walker, SIAM J. Sci. Comput. 17(1), 16-32, 1994
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "optim.h"


/* File handles for PTRN convergence history */
static FILE *_ptrn_fp = NULL;     /* iterate_PTRN.dat */
static FILE *_ptrn_cg_fp = NULL;  /* iterate_PTRN_CG.dat */


/*--------------------------------------------------------------------
 * init_ptrn -- Allocate arrays and initialize PTRN state.
 *
 * Same as init_trn but additionally allocates residual_preco.
 *--------------------------------------------------------------------*/
static void init_ptrn(int n, float *x, float fcost, float *grad,
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

	/* Work arrays */
	opt->xk = (float *)malloc(n * sizeof(float));
	memcpy(opt->xk, x, n * sizeof(float));

	opt->grad = (float *)malloc(n * sizeof(float));
	memcpy(opt->grad, grad, n * sizeof(float));

	opt->descent = (float *)malloc(n * sizeof(float));
	opt->descent_prev = (float *)malloc(n * sizeof(float));
	opt->residual = (float *)malloc(n * sizeof(float));
	opt->residual_preco = (float *)malloc(n * sizeof(float));
	opt->d = (float *)malloc(n * sizeof(float));
	opt->Hd = (float *)malloc(n * sizeof(float));
	opt->eisenvect = (float *)malloc(n * sizeof(float));

	/* Norm of the first gradient */
	opt->norm_grad = optim_norm_l2(n, grad);
}


/*--------------------------------------------------------------------
 * descent_ptrn0 -- CG initialization (preconditioned).
 *
 * r = grad, z = grad_preco, d = -z, descent = 0
 *--------------------------------------------------------------------*/
static void descent_ptrn0(int n, float *grad, float *grad_preco,
                          optim_type *opt, optFlag *flag)
{
	int i;

	memcpy(opt->residual, grad, n * sizeof(float));
	memcpy(opt->residual_preco, grad_preco, n * sizeof(float));
	for (i = 0; i < n; i++) {
		opt->d[i] = -opt->residual_preco[i];
		opt->Hd[i] = 0.0f;
		opt->descent[i] = 0.0f;
	}
	opt->qk_CG = 0.0f;
	opt->hessian_term = 0.0f;
	opt->norm_residual = optim_norm_l2(n, opt->residual);
	opt->conv_CG = 0;
	opt->cpt_iter_CG = 0;

	print_info_ptrn(n, opt, 0.0f, *flag);
}


/*--------------------------------------------------------------------
 * descent_ptrn1 -- CG iteration part 1: before preconditioning.
 *
 * Compute alpha = <r,z>/dHd, update descent and residual.
 * After this, user must apply preconditioner: z = P^{-1} r.
 *--------------------------------------------------------------------*/
static void descent_ptrn1(int n, float *grad, optim_type *opt,
                          optFlag *flag)
{
	int i;
	float dHd, alpha_cg;

	dHd = optim_dot(n, opt->d, opt->Hd);

	if (dHd < 0.0f) {
		/* Negative curvature detected */
		opt->conv_CG = 1;

		if (opt->cpt_iter_CG == 0) {
			/* First iteration: use preconditioned steepest descent */
			memcpy(opt->descent, opt->d, n * sizeof(float));

			if (opt->debug) {
				float grad_term;
				float *mgrad = (float *)malloc(n * sizeof(float));
				for (i = 0; i < n; i++)
					mgrad[i] = -grad[i];
				opt->res_scal_respreco = optim_dot(n, opt->residual,
				                                    opt->residual_preco);
				alpha_cg = opt->res_scal_respreco / dHd;
				opt->qkm1_CG = opt->qk_CG;
				grad_term = optim_dot(n, opt->descent, mgrad);
				opt->hessian_term += alpha_cg * alpha_cg * dHd;
				opt->qk_CG = -grad_term + 0.5f * opt->hessian_term;
				free(mgrad);
			}
		}

	} else {
		/* Normal CG iteration: compute alpha = <r,z>/dHd */
		opt->res_scal_respreco = optim_dot(n, opt->residual,
		                                    opt->residual_preco);
		alpha_cg = opt->res_scal_respreco / dHd;

		/* Update descent direction and residual */
		memcpy(opt->descent_prev, opt->descent, n * sizeof(float));
		for (i = 0; i < n; i++) {
			opt->descent[i] += alpha_cg * opt->d[i];
			opt->residual[i] += alpha_cg * opt->Hd[i];
		}
		/* Store alpha_cg for debug quad model in ptrn2 */
		opt->alpha_CG = alpha_cg;
		opt->dHd = dHd;
		/* STOP HERE: user must apply preconditioner to residual */
	}
}


/*--------------------------------------------------------------------
 * descent_ptrn2 -- CG iteration part 2: after preconditioning.
 *
 * User has set residual_preco = P^{-1} * residual.
 * Compute beta = <r_new, z_new> / <r_old, z_old>, update d.
 *--------------------------------------------------------------------*/
static void descent_ptrn2(int n, float *grad, optim_type *opt,
                          optFlag *flag)
{
	int i;
	float res_scal_respreco_prev, beta;

	/* beta = <r_new, z_new> / <r_old, z_old> */
	res_scal_respreco_prev = opt->res_scal_respreco;
	opt->res_scal_respreco = optim_dot(n, opt->residual,
	                                    opt->residual_preco);
	beta = opt->res_scal_respreco / res_scal_respreco_prev;

	/* Update CG direction: d = -z + beta*d */
	for (i = 0; i < n; i++)
		opt->d[i] = -opt->residual_preco[i] + beta * opt->d[i];

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
		opt->hessian_term += opt->alpha_CG * opt->alpha_CG * opt->dHd +
		                     2.0f * opt->alpha_CG * descent_scal_Hd;
		opt->qk_CG = -grad_term + 0.5f * opt->hessian_term;
		free(mgrad);
	}

	/* Check Eisenstat-Walker stopping criterion */
	opt->norm_residual = optim_norm_l2(n, opt->residual);
	opt->conv_CG = (opt->norm_residual <= opt->eta * opt->norm_grad) ||
	               (opt->cpt_iter_CG >= opt->niter_max_CG);

	print_info_ptrn(n, opt, 0.0f, *flag);
}


/*--------------------------------------------------------------------
 * finalize_ptrn -- Free PTRN arrays.
 *--------------------------------------------------------------------*/
static void finalize_ptrn(optim_type *opt)
{
	if (opt->xk) { free(opt->xk); opt->xk = NULL; }
	if (opt->grad) { free(opt->grad); opt->grad = NULL; }
	if (opt->descent) { free(opt->descent); opt->descent = NULL; }
	if (opt->descent_prev) { free(opt->descent_prev); opt->descent_prev = NULL; }
	if (opt->residual) { free(opt->residual); opt->residual = NULL; }
	if (opt->residual_preco) { free(opt->residual_preco); opt->residual_preco = NULL; }
	if (opt->d) { free(opt->d); opt->d = NULL; }
	if (opt->Hd) { free(opt->Hd); opt->Hd = NULL; }
	if (opt->eisenvect) { free(opt->eisenvect); opt->eisenvect = NULL; }
}


/*--------------------------------------------------------------------
 * print_info_ptrn -- Write PTRN convergence history to files.
 *--------------------------------------------------------------------*/
void print_info_ptrn(int n, optim_type *opt, float fcost, optFlag flag)
{
	if (!opt->print_flag) return;

	if (flag == OPT_INIT) {
		_ptrn_fp = fopen("iterate_PTRN.dat", "w");
		if (_ptrn_fp) {
			fprintf(_ptrn_fp,
				"******************************************************************************************\n");
			fprintf(_ptrn_fp,
				"                           PRECONDITIONED TRUNCATED NEWTON ALGORITHM\n");
			fprintf(_ptrn_fp,
				"******************************************************************************************\n");
			fprintf(_ptrn_fp, "     Convergence criterion  : %10.2e\n", opt->conv);
			fprintf(_ptrn_fp, "     Niter_max              : %7d\n", opt->niter_max);
			fprintf(_ptrn_fp, "     Initial cost is        : %10.2e\n", opt->f0);
			fprintf(_ptrn_fp, "     Initial norm_grad is   : %10.2e\n", opt->norm_grad);
			fprintf(_ptrn_fp, "     Maximum CG iter        : %7d\n", opt->niter_max_CG);
			fprintf(_ptrn_fp,
				"******************************************************************************************\n");
			fprintf(_ptrn_fp,
				"   Niter        fk      ||gk||      fk/f0"
				"         alpha       nls   nit_CG"
				"         eta    ngrad   nhess\n");
			fprintf(_ptrn_fp,
				"%6d%12.2e%12.2e%12.2e%12.2e%8d%8d%12.2e%8d%8d\n",
				opt->cpt_iter, fcost, opt->norm_grad,
				(opt->f0 > 0.0f) ? fcost / opt->f0 : 0.0f,
				opt->alpha, opt->cpt_ls, opt->cpt_iter_CG,
				opt->eta, opt->nfwd_pb, opt->nhess);
			fflush(_ptrn_fp);
		}

		_ptrn_cg_fp = fopen("iterate_PTRN_CG.dat", "w");
		if (_ptrn_cg_fp) {
			fprintf(_ptrn_cg_fp,
				"******************************************************************************************\n");
			fprintf(_ptrn_cg_fp,
				"                           PRECONDITIONED TRUNCATED NEWTON ALGORITHM\n");
			fprintf(_ptrn_cg_fp,
				"                                      INNER CG HISTORY\n");
			fprintf(_ptrn_cg_fp,
				"******************************************************************************************\n");
			fprintf(_ptrn_cg_fp, "     Convergence criterion  : %10.2e\n", opt->conv);
			fprintf(_ptrn_cg_fp, "     Niter_max              : %7d\n", opt->niter_max);
			fprintf(_ptrn_cg_fp, "     Initial cost is        : %10.2e\n", opt->f0);
			fprintf(_ptrn_cg_fp, "     Initial norm_grad is   : %10.2e\n", opt->norm_grad);
			fprintf(_ptrn_cg_fp, "     Maximum CG iter        : %7d\n", opt->niter_max_CG);
			fprintf(_ptrn_cg_fp,
				"******************************************************************************************\n");
			fflush(_ptrn_cg_fp);
		}

	} else if (flag == OPT_CONV) {
		if (_ptrn_fp) {
			fprintf(_ptrn_fp,
				"**********************************************************************\n");
			if (opt->cpt_iter >= opt->niter_max)
				fprintf(_ptrn_fp,
					"  STOP: MAXIMUM NUMBER OF ITERATION REACHED\n");
			else
				fprintf(_ptrn_fp,
					"  STOP: CONVERGENCE CRITERION SATISFIED\n");
			fprintf(_ptrn_fp,
				"**********************************************************************\n");
			fclose(_ptrn_fp);
			_ptrn_fp = NULL;
		}
		if (_ptrn_cg_fp) {
			fclose(_ptrn_cg_fp);
			_ptrn_cg_fp = NULL;
		}

	} else if (flag == OPT_FAIL) {
		if (_ptrn_fp) {
			fprintf(_ptrn_fp,
				"**********************************************************************\n");
			fprintf(_ptrn_fp,
				"  STOP: LINESEARCH FAILURE\n");
			fprintf(_ptrn_fp,
				"**********************************************************************\n");
			fclose(_ptrn_fp);
			_ptrn_fp = NULL;
		}
		if (_ptrn_cg_fp) {
			fclose(_ptrn_cg_fp);
			_ptrn_cg_fp = NULL;
		}

	} else if (opt->comm == TRN_DESC) {
		/* CG iteration info */
		if (_ptrn_cg_fp) {
			if (opt->CG_phase == CG_INIT || opt->cpt_iter_CG == 0) {
				fprintf(_ptrn_cg_fp,
					"-------------------------------------------------------------------------------------------------\n");
				fprintf(_ptrn_cg_fp,
					" NONLINEAR ITERATION %4d ETA IS : %12.2e\n",
					opt->cpt_iter, opt->eta);
				fprintf(_ptrn_cg_fp,
					"-------------------------------------------------------------------------------------------------\n");
				fprintf(_ptrn_cg_fp,
					"  Iter_CG           qk     norm_res   norm_res/||gk||\n");
				fprintf(_ptrn_cg_fp,
					"%8d%12.2e%12.2e%12.2e\n",
					opt->cpt_iter_CG, opt->qk_CG,
					opt->norm_residual,
					(opt->norm_grad > 0.0f) ?
					opt->norm_residual / opt->norm_grad : 0.0f);
			} else {
				fprintf(_ptrn_cg_fp,
					"%8d%12.2e%12.2e%12.2e\n",
					opt->cpt_iter_CG, opt->qk_CG,
					opt->norm_residual,
					(opt->norm_grad > 0.0f) ?
					opt->norm_residual / opt->norm_grad : 0.0f);
			}
			fflush(_ptrn_cg_fp);
		}

	} else {
		/* Outer iteration info (from linesearch) */
		if (_ptrn_fp) {
			fprintf(_ptrn_fp,
				"%6d%12.2e%12.2e%12.2e%12.2e%8d%8d%12.2e%8d%8d\n",
				opt->cpt_iter, fcost, opt->norm_grad,
				(opt->f0 > 0.0f) ? fcost / opt->f0 : 0.0f,
				opt->alpha, opt->cpt_ls, opt->cpt_iter_CG,
				opt->eta, opt->nfwd_pb, opt->nhess);
			fflush(_ptrn_fp);
		}
	}
}


/*--------------------------------------------------------------------
 * ptrn_run -- Main PTRN reverse communication dispatcher.
 *
 * Like trn_run but with preconditioned CG inner loop.
 * The CG iteration is split into two phases with a preconditioning
 * step in between:
 *   DES1: compute Hd, then alpha/update (descent_ptrn1)
 *         -> returns OPT_PREC (user sets residual_preco = P^{-1}*residual)
 *   DES2: compute beta, update CG direction (descent_ptrn2)
 *         -> returns OPT_HESS (user computes Hd = H*d)
 *
 * Returns:
 *   OPT_GRAD: user must compute cost, gradient, AND preconditioned gradient
 *   OPT_HESS: user must compute opt->Hd = H * opt->d
 *   OPT_PREC: user must compute opt->residual_preco = P^{-1} * opt->residual
 *   OPT_NSTE: new step accepted
 *   OPT_CONV: converged
 *   OPT_FAIL: linesearch failed
 *--------------------------------------------------------------------*/
void ptrn_run(int n, float *x, float fcost, float *grad,
              float *grad_preco, optim_type *opt, optFlag *flag)
{
	if (*flag == OPT_INIT) {
		/*--- Initialization ---*/
		init_ptrn(n, x, fcost, grad, opt);
		print_info_ptrn(n, opt, fcost, OPT_INIT);
		opt->comm = TRN_DESC;
		opt->CG_phase = CG_INIT;
		opt->nfwd_pb++;
		opt->conv_CG = 0;
		*flag = OPT_INIT;  /* Will fall through to DESC below */
	}

	if (opt->comm == TRN_DESC) {
		/*--- Computing descent direction via preconditioned CG ---*/
		if (opt->CG_phase == CG_INIT) {
			/* CG initialization */
			descent_ptrn0(n, grad, grad_preco, opt, flag);
			opt->CG_phase = CG_IRUN;
			opt->comm = TRN_DESC;
			*flag = OPT_HESS;
			opt->nhess++;

		} else {
			/* CG iteration part 1: process Hd, update residual */
			descent_ptrn1(n, grad, opt, flag);

			if (opt->conv_CG) {
				/* CG converged (negative curvature): go to linesearch */
				opt->comm = TRN_NSTE;
				opt->CG_phase = CG_INIT;

				optim_wolfe_linesearch(n, x, fcost, grad, opt);
				*flag = OPT_GRAD;
				opt->nfwd_pb++;
			} else {
				/* Ask user to precondition the residual */
				*flag = OPT_PREC;
				opt->comm = PTRN_DES2;
			}
		}

	} else if (opt->comm == PTRN_DES2) {
		/*--- CG iteration part 2: after preconditioning ---*/
		descent_ptrn2(n, grad, opt, flag);

		if (opt->conv_CG) {
			/* CG converged: go to linesearch */
			opt->comm = TRN_NSTE;
			opt->CG_phase = CG_INIT;

			optim_wolfe_linesearch(n, x, fcost, grad, opt);
			*flag = OPT_GRAD;
			opt->nfwd_pb++;
		} else {
			/* Need another Hessian-vector product */
			opt->comm = TRN_DESC;
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
			print_info_ptrn(n, opt, fcost, OPT_NSTE);

			if (optim_test_conv(opt, fcost)) {
				*flag = OPT_CONV;
				print_info_ptrn(n, opt, fcost, OPT_CONV);
				finalize_ptrn(opt);
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
			print_info_ptrn(n, opt, fcost, OPT_FAIL);
			finalize_ptrn(opt);
		}
	}
}
