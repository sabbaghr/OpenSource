/*
 * penriched.c - Preconditioned Enriched: PLBFGS + PTRN hybrid.
 *
 * Extension of the Enriched Method (Morales & Nocedal 2002):
 *   - L-BFGS phase upgraded to PLBFGS: two-loop uses external P^{-1} as H0
 *   - HFN phase inner CG uses doubly-preconditioned L-BFGS: M = P * H_lbfgs
 *
 * The L-BFGS two-loop recursion is split into backward/forward halves.
 * Between them, OPT_PREC returns to the caller to apply P^{-1} to q_plb.
 * This replaces the identity/gamma initial Hessian approximation with the
 * user's preconditioner (source illumination diagonal in FWI).
 *
 * Returns OPT_PREC when user must apply P^{-1} to opt->q_plb.
 * Returns OPT_HESS when user must compute opt->Hd = H * opt->d.
 *
 * Reference: Morales & Nocedal, Comput. Optim. Appl., 21, 143-154, 2002
 *            Metivier et al., SIAM Review, 59(1), 153-195, 2017
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "optim.h"


/* File handle for convergence history */
static FILE *_penr_fp = NULL;


/*--------------------------------------------------------------------
 * save_lbfgs_pair -- Store a (s,y) pair directly into L-BFGS buffer.
 *
 * Used for curvature seeding: stores inner CG pairs (d, Hd).
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
 * lbfgs_backward_loop -- First half of L-BFGS two-loop recursion.
 *
 * Computes q_plb from input via backward pass through history.
 * Allocates alpha_plb, rho_plb (freed by lbfgs_forward_loop).
 * Returns borne (number of history pairs, 0 = no history).
 *--------------------------------------------------------------------*/
static int lbfgs_backward_loop(int n, optim_type *opt, const float *input)
{
	int i, idx, borne;

	borne = opt->cpt_lbfgs - 1;

	/* q_plb = input */
	memcpy(opt->q_plb, input, n * sizeof(float));

	if (borne <= 0)
		return 0;

	/* Check norms of most recent pair */
	{
		float norm_sk = optim_norm_l2(n, &opt->sk[(borne - 1) * n]);
		float norm_yk = optim_norm_l2(n, &opt->yk[(borne - 1) * n]);
		if (norm_sk == 0.0f || norm_yk == 0.0f)
			return 0;
	}

	/* Allocate per-call work arrays */
	opt->alpha_plb = (float *)malloc(borne * sizeof(float));
	opt->rho_plb = (float *)malloc(borne * sizeof(float));

	/* Backward loop through history (newest to oldest) */
	for (i = 0; i < borne; i++) {
		idx = borne - 1 - i;

		opt->rho_plb[idx] = 1.0f / optim_dot(n, &opt->yk[idx * n],
		                                       &opt->sk[idx * n]);
		opt->alpha_plb[idx] = opt->rho_plb[idx] *
		                       optim_dot(n, &opt->sk[idx * n], opt->q_plb);

		for (int j = 0; j < n; j++)
			opt->q_plb[j] -= opt->alpha_plb[idx] * opt->yk[idx * n + j];
	}
	return borne;
}


/*--------------------------------------------------------------------
 * lbfgs_forward_loop -- Second half of L-BFGS two-loop recursion.
 *
 * Called after user has applied P^{-1} to q_plb.
 * Computes output = -H_precond * original_input.
 * If borne_saved == 0, just negates q_plb (identity + precond).
 * Frees alpha_plb, rho_plb.
 *
 * borne_saved: value returned by lbfgs_backward_loop.
 *--------------------------------------------------------------------*/
static void lbfgs_forward_loop(int n, optim_type *opt, float *output,
                               int borne_saved)
{
	int i, idx;

	if (borne_saved <= 0) {
		/* No history: output = -P^{-1}*input = -q_plb */
		for (i = 0; i < n; i++)
			output[i] = -opt->q_plb[i];
		return;
	}

	/* Gamma scaling from most recent pair */
	{
		float gamma_num, gamma_den, gamma;
		gamma_num = optim_dot(n, &opt->sk[(borne_saved - 1) * n],
		                       &opt->yk[(borne_saved - 1) * n]);
		gamma_den = optim_norm_l2(n, &opt->yk[(borne_saved - 1) * n]);
		gamma = gamma_num / (gamma_den * gamma_den);

		for (i = 0; i < n; i++)
			output[i] = gamma * opt->q_plb[i];
	}

	/* Forward loop through history (oldest to newest) */
	for (idx = 0; idx < borne_saved; idx++) {
		float beta = opt->rho_plb[idx] *
		             optim_dot(n, &opt->yk[idx * n], output);
		for (i = 0; i < n; i++)
			output[i] += (opt->alpha_plb[idx] - beta) *
			              opt->sk[idx * n + i];
	}

	/* Negate: output = -H_precond * input */
	for (i = 0; i < n; i++)
		output[i] = -output[i];

	/* Free per-call work arrays */
	free(opt->alpha_plb); opt->alpha_plb = NULL;
	free(opt->rho_plb); opt->rho_plb = NULL;
}


/*--------------------------------------------------------------------
 * init_penriched -- Allocate arrays and initialize state.
 *--------------------------------------------------------------------*/
static void init_penriched(int n, float *x, float fcost, float *grad,
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
	opt->q_plb = (float *)malloc(n * sizeof(float));

	/* Gradient norm */
	opt->norm_grad = optim_norm_l2(n, grad);

	/* Enriched state: start in HFN mode */
	opt->enr_method = 1;
	opt->enr_first = 1;
	if (opt->enr_l <= 0) opt->enr_l = 20;
	opt->enr_t = 2;
	opt->enr_k = 0;
	opt->enr_profit = 0;
	opt->enr_force2 = 0;
	if (opt->enr_maxcg <= 0) opt->enr_maxcg = 5;
	opt->enr_comm = TRN_DESC;
	opt->enr_recovering = 0;

	/* borne_saved stored in niter_max_CG temporarily during two-loop split */
}


/*--------------------------------------------------------------------
 * forcing_term_penriched -- Eisenstat-Walker forcing term.
 *--------------------------------------------------------------------*/
static void forcing_term_penriched(int n, float *grad, optim_type *opt)
{
	int i;
	float eta_save, eta_save_power, norm_eisenvect;

	eta_save = opt->eta;

	for (i = 0; i < n; i++)
		opt->eisenvect[i] = grad[i] - opt->residual[i];

	norm_eisenvect = optim_norm_l2(n, opt->eisenvect);
	opt->eta = norm_eisenvect / opt->norm_grad_m1;

	eta_save_power = powf(eta_save, (1.0f + sqrtf(5.0f)) / 2.0f);
	if (eta_save_power > 0.1f)
		opt->eta = fmaxf(opt->eta, eta_save_power);

	if (opt->eta > 1.0f)
		opt->eta = 0.9f;
}


/*--------------------------------------------------------------------
 * adjust_penriched -- ADJUST procedure (Morales & Nocedal 2002).
 *--------------------------------------------------------------------*/
static void adjust_penriched(optim_type *opt)
{
	opt->enr_k++;

	if (opt->enr_method == 0) {
		/* L-BFGS phase */
		if (opt->enr_k >= opt->enr_l) {
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
			opt->enr_profit++;
		} else {
			if (opt->enr_force2 && opt->enr_k == 1)
				return;
			opt->enr_t = (opt->enr_k - 1 > 2) ? opt->enr_k - 1 : 2;
			opt->enr_method = 0;
			opt->enr_k = 0;
			return;
		}

		if (opt->enr_k >= opt->enr_t) {
			if (opt->enr_profit == opt->enr_k)
				opt->enr_t++;
			opt->enr_force2 = (opt->enr_profit >= 2) ? 1 : 0;
			opt->enr_method = 0;
			opt->enr_k = 0;
		}
	}
}


/*--------------------------------------------------------------------
 * print_info_penriched -- Write convergence history.
 *--------------------------------------------------------------------*/
static void print_info_penriched(int n, optim_type *opt, float fcost,
                                 optFlag flag)
{
	if (!opt->print_flag) return;

	if (flag == OPT_INIT) {
		_penr_fp = fopen("iterate_PENR.dat", "w");
		if (!_penr_fp) return;

		fprintf(_penr_fp,
			"**********************************************************************\n");
		fprintf(_penr_fp,
			"        PRECONDITIONED ENRICHED OPTIMIZATION (PLBFGS + PTRN)\n");
		fprintf(_penr_fp,
			"**********************************************************************\n");
		fprintf(_penr_fp, "     Convergence criterion  : %10.2e\n", opt->conv);
		fprintf(_penr_fp, "     Niter_max              : %7d\n", opt->niter_max);
		fprintf(_penr_fp, "     Initial cost is        : %10.2e\n", opt->f0);
		fprintf(_penr_fp, "     Initial norm_grad is   : %10.2e\n", opt->norm_grad);
		fprintf(_penr_fp, "     L-BFGS cycle length    : %7d\n", opt->enr_l);
		fprintf(_penr_fp, "     Memory parameter m     : %7d\n", opt->l);
		fprintf(_penr_fp, "     Initial maxcg          : %7d\n", opt->enr_maxcg);
		fprintf(_penr_fp,
			"**********************************************************************\n");
		fprintf(_penr_fp,
			"   Niter        fk      ||gk||      fk/f0"
			"         alpha  method   nls   nit_CG"
			"         eta    ngrad   nhess\n");
		fprintf(_penr_fp,
			"%6d%12.2e%12.2e%12.2e%12.2e    %s%8d%8d%12.2e%8d%8d\n",
			opt->cpt_iter, fcost, opt->norm_grad,
			1.0f, 0.0f,
			opt->enr_method ? "HFN" : "PLB",
			opt->cpt_ls, opt->cpt_iter_CG,
			opt->eta, opt->nfwd_pb, opt->nhess);
		fflush(_penr_fp);

	} else if (flag == OPT_CONV) {
		if (!_penr_fp) return;
		fprintf(_penr_fp,
			"**********************************************************************\n");
		if (opt->cpt_iter >= opt->niter_max)
			fprintf(_penr_fp,
				"  STOP: MAXIMUM NUMBER OF ITERATION REACHED\n");
		else
			fprintf(_penr_fp,
				"  STOP: CONVERGENCE CRITERION SATISFIED\n");
		fprintf(_penr_fp,
			"**********************************************************************\n");
		fclose(_penr_fp);
		_penr_fp = NULL;

	} else if (flag == OPT_FAIL) {
		if (!_penr_fp) return;
		fprintf(_penr_fp,
			"**********************************************************************\n");
		fprintf(_penr_fp, "  STOP: LINESEARCH FAILURE\n");
		fprintf(_penr_fp,
			"**********************************************************************\n");
		fclose(_penr_fp);
		_penr_fp = NULL;

	} else {
		if (!_penr_fp) return;
		fprintf(_penr_fp,
			"%6d%12.2e%12.2e%12.2e%12.2e    %s%8d%8d%12.2e%8d%8d\n",
			opt->cpt_iter, fcost, opt->norm_grad,
			(opt->f0 > 0.0f) ? fcost / opt->f0 : 0.0f,
			opt->alpha,
			opt->enr_method ? "HFN" : "PLB",
			opt->cpt_ls, opt->cpt_iter_CG,
			opt->eta, opt->nfwd_pb, opt->nhess);
		fflush(_penr_fp);
	}
}


/*--------------------------------------------------------------------
 * finalize_penriched -- Free all arrays.
 *--------------------------------------------------------------------*/
static void finalize_penriched(optim_type *opt)
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
	if (opt->q_plb) { free(opt->q_plb); opt->q_plb = NULL; }
	if (opt->alpha_plb) { free(opt->alpha_plb); opt->alpha_plb = NULL; }
	if (opt->rho_plb) { free(opt->rho_plb); opt->rho_plb = NULL; }
}


/*--------------------------------------------------------------------
 * penriched_run -- Main preconditioned enriched dispatcher.
 *
 * State machine via enr_comm:
 *   TRN_DESC: HFN inner CG (process Hd or start CG)
 *   ENR_PREC: waiting for P^{-1} on q_plb
 *     + enr_method==0: L-BFGS descent → OPT_NSTE (or OPT_GRAD if recovering)
 *     + enr_method==1, CG_INIT: CG init z → OPT_HESS
 *     + enr_method==1, CG_IRUN: CG step z → OPT_HESS or linesearch
 *   TRN_NSTE: linesearch
 *
 * Returns: OPT_GRAD, OPT_HESS, OPT_PREC, OPT_NSTE, OPT_CONV, OPT_FAIL
 *--------------------------------------------------------------------*/
void penriched_run(int n, float *x, float fcost, float *grad,
                   float *grad_preco, optim_type *opt, optFlag *flag)
{
	int i, borne;

	if (*flag == OPT_INIT) {
		/*--- Initialization ---*/
		init_penriched(n, x, fcost, grad, opt);
		print_info_penriched(n, opt, fcost, OPT_INIT);
		opt->nfwd_pb++;

		/* Start in HFN mode: begin CG via preconditioned two-loop */
		opt->enr_comm = TRN_DESC;
		opt->CG_phase = CG_INIT;
		opt->conv_CG = 0;
		/* Fall through to TRN_DESC handler */
	}

	/* ================================================================
	 * TRN_DESC: HFN inner CG iteration
	 * ================================================================ */
	if (opt->enr_comm == TRN_DESC && opt->enr_method == 1) {

		if (opt->CG_phase == CG_INIT) {
			/*--- CG initialization ---*/
			memcpy(opt->residual, grad, n * sizeof(float));
			for (i = 0; i < n; i++) {
				opt->Hd[i] = 0.0f;
				opt->descent[i] = 0.0f;
			}
			opt->norm_residual = optim_norm_l2(n, opt->residual);
			opt->conv_CG = 0;
			opt->cpt_iter_CG = 0;

			/* Backward loop on residual → q_plb → OPT_PREC */
			borne = lbfgs_backward_loop(n, opt, opt->residual);
			opt->niter_max_CG = borne;  /* save borne for forward */
			opt->enr_comm = ENR_PREC;
			opt->CG_phase = CG_INIT;  /* keep CG_INIT to identify this is init */
			*flag = OPT_PREC;
			return;

		} else {
			/*--- CG iteration: process Hd ---*/
			float dHd = optim_dot(n, opt->d, opt->Hd);

			if (dHd <= 0.0f) {
				/* Negative curvature: stop CG */
				opt->conv_CG = 1;
				if (opt->cpt_iter_CG == 0)
					memcpy(opt->descent, opt->d, n * sizeof(float));
				/* Go to linesearch */
				opt->enr_comm = TRN_NSTE;
				opt->CG_phase = CG_INIT;
				optim_wolfe_linesearch(n, x, fcost, grad, opt);
				*flag = OPT_GRAD;
				opt->nfwd_pb++;
				return;
			}

			/* Store (d, Hd) curvature pair in L-BFGS buffer */
			{
				float norm_d = optim_norm_l2(n, opt->d);
				if (norm_d > 0.0f)
					save_lbfgs_pair(n, opt, opt->d, opt->Hd);
			}

			/* rho_old = <r, z> = -<r, enr_precond_z> (since z = -enr_precond_z) */
			float rho_old = -optim_dot(n, opt->residual, opt->enr_precond_z);
			float alpha_cg = rho_old / dHd;

			memcpy(opt->descent_prev, opt->descent, n * sizeof(float));
			for (i = 0; i < n; i++) {
				opt->descent[i] += alpha_cg * opt->d[i];
				opt->residual[i] += alpha_cg * opt->Hd[i];
			}

			opt->cpt_iter_CG++;
			opt->norm_residual = optim_norm_l2(n, opt->residual);
			/* Save rho_old for beta computation after preconditioner */
			opt->res_scal_respreco = rho_old;

			/* Check stopping criterion */
			if (opt->norm_residual <= opt->eta * opt->norm_grad ||
			    opt->cpt_iter_CG >= opt->enr_maxcg) {
				opt->conv_CG = 1;
				/* Go to linesearch */
				opt->enr_comm = TRN_NSTE;
				opt->CG_phase = CG_INIT;
				optim_wolfe_linesearch(n, x, fcost, grad, opt);
				*flag = OPT_GRAD;
				opt->nfwd_pb++;
				return;
			}

			/* Need preconditioned z_new: backward loop → OPT_PREC */
			borne = lbfgs_backward_loop(n, opt, opt->residual);
			opt->niter_max_CG = borne;  /* save borne */
			opt->enr_comm = ENR_PREC;
			opt->CG_phase = CG_IRUN;
			*flag = OPT_PREC;
			return;
		}
	}

	/* ================================================================
	 * ENR_PREC: user has applied P^{-1} to q_plb
	 * ================================================================ */
	if (opt->enr_comm == ENR_PREC) {
		borne = opt->niter_max_CG;  /* recover saved borne */

		if (opt->enr_method == 0) {
			/*--- L-BFGS phase: complete descent computation ---*/
			lbfgs_forward_loop(n, opt, opt->descent, borne);

			if (opt->enr_recovering) {
				/* Recovering from HFN failure: need fresh gradient */
				opt->enr_recovering = 0;
				opt->first_ls = 1;
				opt->enr_comm = TRN_NSTE;
				*flag = OPT_GRAD;
				opt->nfwd_pb++;
			} else {
				/* Normal step: descent ready, signal iteration complete */
				opt->enr_comm = TRN_NSTE;
				*flag = OPT_NSTE;
			}
			return;

		} else if (opt->CG_phase == CG_INIT) {
			/*--- HFN CG init: complete z0 computation ---*/
			lbfgs_forward_loop(n, opt, opt->enr_precond_z, borne);
			/* enr_precond_z = -H_precond * r = -z0 */
			/* d = -z0 (which equals enr_precond_z) */
			memcpy(opt->d, opt->enr_precond_z, n * sizeof(float));
			opt->CG_phase = CG_IRUN;
			opt->enr_comm = TRN_DESC;
			*flag = OPT_HESS;
			opt->nhess++;
			return;

		} else {
			/*--- HFN CG step: complete z_new, compute beta ---*/
			lbfgs_forward_loop(n, opt, opt->enr_precond_z, borne);
			/* enr_precond_z = -z_new */

			/* rho_new = <r_new, z_new> = -<r_new, enr_precond_z> */
			float rho_new = -optim_dot(n, opt->residual, opt->enr_precond_z);
			float rho_old = opt->res_scal_respreco;  /* saved before OPT_PREC */
			float beta = rho_new / rho_old;

			/* d = -z_new + beta * d = enr_precond_z + beta * d */
			for (i = 0; i < n; i++)
				opt->d[i] = opt->enr_precond_z[i] + beta * opt->d[i];

			opt->enr_comm = TRN_DESC;
			*flag = OPT_HESS;
			opt->nhess++;
			return;
		}
	}

	/* ================================================================
	 * TRN_NSTE: linesearch phase (both L-BFGS and HFN)
	 * ================================================================ */
	optim_wolfe_linesearch(n, x, fcost, grad, opt);

	if (opt->ls_task == LS_NEW_STEP) {
		/* Step accepted */
		int prev_method = opt->enr_method;

		opt->cpt_iter++;
		opt->norm_grad_m1 = opt->norm_grad;
		opt->norm_grad = optim_norm_l2(n, grad);

		/* Store outer (s,y) pair in L-BFGS buffer */
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

		memcpy(opt->grad, grad, n * sizeof(float));
		print_info_penriched(n, opt, fcost, OPT_NSTE);

		if (optim_test_conv(opt, fcost)) {
			*flag = OPT_CONV;
			print_info_penriched(n, opt, fcost, OPT_CONV);
			finalize_penriched(opt);
			return;
		}

		/* Handle indefinite Hessian: force L-BFGS */
		if (prev_method == 1 && opt->cpt_iter_CG == 0) {
			opt->enr_t = 1;
			opt->enr_force2 = 0;
			opt->enr_l = (3 * opt->enr_l / 2 < 30) ?
			              3 * opt->enr_l / 2 : 30;
			opt->enr_method = 0;
			opt->enr_k = 0;
		} else {
			adjust_penriched(opt);
		}

		/* Forcing term (only valid after HFN) */
		if (prev_method == 1 && opt->cpt_iter > 1)
			forcing_term_penriched(n, grad, opt);

		if (opt->enr_method == 0) {
			/*--- Switching to / continuing L-BFGS phase ---*/
			/* Split two-loop: backward → OPT_PREC → forward → descent */
			borne = lbfgs_backward_loop(n, opt, grad);
			opt->niter_max_CG = borne;  /* save borne */
			opt->enr_comm = ENR_PREC;
			opt->enr_recovering = 0;
			*flag = OPT_PREC;
		} else {
			/*--- Switching to / continuing HFN phase ---*/
			/* CG will start on next call via TRN_DESC + CG_INIT */
			opt->enr_comm = TRN_DESC;
			opt->CG_phase = CG_INIT;
			opt->conv_CG = 0;
			*flag = OPT_NSTE;
		}

	} else if (opt->ls_task == LS_NEW_GRAD) {
		*flag = OPT_GRAD;
		opt->nfwd_pb++;

	} else if (opt->ls_task == LS_FAILURE) {
		if (opt->enr_method == 1) {
			/* HFN failure: switch to L-BFGS, recover */
			opt->enr_method = 0;
			opt->enr_k = 0;
			opt->enr_t = 1;
			opt->enr_l = (3 * opt->enr_l / 2 < 30) ?
			              3 * opt->enr_l / 2 : 30;

			memcpy(x, opt->xk, n * sizeof(float));

			/* Split two-loop on saved gradient → OPT_PREC */
			borne = lbfgs_backward_loop(n, opt, opt->grad);
			opt->niter_max_CG = borne;
			opt->enr_comm = ENR_PREC;
			opt->enr_recovering = 1;
			*flag = OPT_PREC;
		} else {
			/* L-BFGS failure: truly done */
			*flag = OPT_FAIL;
			memcpy(opt->grad, grad, n * sizeof(float));
			print_info_penriched(n, opt, fcost, OPT_FAIL);
			finalize_penriched(opt);
		}
	}
}
