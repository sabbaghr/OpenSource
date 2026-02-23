/*
 * test_rosenbrock.c - Unit tests for all optimization algorithms.
 *
 * Minimizes the 2D Rosenbrock function:
 *   f(x,y) = (1-x)^2 + 100*(y-x^2)^2
 *   Minimum at (1,1), f=0.
 *
 * Tests:
 *   1. L-BFGS converges to (1,1) from (1.5, 1.5)
 *   2. Steepest descent converges (same start, more iterations)
 *   3. L-BFGS with bound constraints [0,2]x[0,2]
 *   4. PNLCG (Dai-Yuan) converges to (1,1)
 *   5. PLBFGS converges to (1,1)
 *   6. TRN (Truncated Newton) converges to (1,1)
 *
 * Same test as TOOLBOX_OPTIMIZATION/LBFGS/test/src/test_LBFGS.f90
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "optim.h"

#define TOL 2e-3f  /* matches Fortran: final x~(1.0004, 1.0008) for conv=1e-8 */


/*--------------------------------------------------------------------
 * rosenbrock -- Compute cost and gradient.
 *
 * f(x1,x2)  = (1-x1)^2 + 100*(x2-x1^2)^2
 * df/dx1    = 2*(x1-1) - 400*x1*(x2-x1^2)
 * df/dx2    = 200*(x2-x1^2)
 *--------------------------------------------------------------------*/
static void rosenbrock(float *x, float *fcost, float *grad)
{
	float x1 = x[0], x2 = x[1];
	float t = x2 - x1 * x1;

	*fcost = (1.0f - x1) * (1.0f - x1) + 100.0f * t * t;
	grad[0] = 2.0f * (x1 - 1.0f) - 400.0f * x1 * t;
	grad[1] = 200.0f * t;
}


/*--------------------------------------------------------------------
 * rosenbrock_hessian_vec -- Hessian-vector product for Rosenbrock.
 *
 * H(x) = [ 2 + 1200*x1^2 - 400*x2,  -400*x1 ]
 *         [ -400*x1,                   200     ]
 *
 * Hd[0] = H[0][0]*d[0] + H[0][1]*d[1]
 * Hd[1] = H[1][0]*d[0] + H[1][1]*d[1]
 *--------------------------------------------------------------------*/
static void rosenbrock_hessian_vec(float *x, float *d, float *Hd)
{
	float x1 = x[0], x2 = x[1];
	float h00 = 2.0f + 1200.0f * x1 * x1 - 400.0f * x2;
	float h01 = -400.0f * x1;
	float h11 = 200.0f;

	Hd[0] = h00 * d[0] + h01 * d[1];
	Hd[1] = h01 * d[0] + h11 * d[1];
}


/*--------------------------------------------------------------------
 * test_lbfgs -- Test L-BFGS on Rosenbrock.
 *--------------------------------------------------------------------*/
static int test_lbfgs(void)
{
	int n = 2;
	float x[2] = {1.5f, 1.5f};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;

	printf("=== TEST 1: L-BFGS on Rosenbrock ===\n");

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 10000;
	opt.conv = 1e-8f;
	opt.print_flag = 1;
	opt.debug = 0;
	opt.l = 20;

	rosenbrock(x, &fcost, grad);
	printf("  Initial: x=(%.4f, %.4f), f=%.6e\n", x[0], x[1], fcost);

	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		lbfgs_run(n, x, fcost, grad, &opt, &flag);
		if (flag == OPT_GRAD)
			rosenbrock(x, &fcost, grad);
	}

	printf("  Final:   x=(%.6f, %.6f), f=%.6e\n", x[0], x[1], fcost);
	printf("  Iterations: %d, Forward problems: %d\n",
	       opt.cpt_iter, opt.nfwd_pb);

	if (flag == OPT_FAIL) {
		printf("  FAIL: linesearch failed\n");
		return 1;
	}
	if (fabsf(x[0] - 1.0f) > TOL || fabsf(x[1] - 1.0f) > TOL) {
		printf("  FAIL: did not converge to (1,1)\n");
		return 1;
	}

	printf("  PASS\n\n");
	return 0;
}


/*--------------------------------------------------------------------
 * test_sd -- Test steepest descent on Rosenbrock.
 *--------------------------------------------------------------------*/
static int test_sd(void)
{
	int n = 2;
	float x[2] = {1.5f, 1.5f};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;

	printf("=== TEST 2: Steepest descent on Rosenbrock ===\n");

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 50000;
	opt.conv = 1e-5f;  /* Looser tolerance for SD */
	opt.print_flag = 1;
	opt.debug = 0;

	rosenbrock(x, &fcost, grad);
	printf("  Initial: x=(%.4f, %.4f), f=%.6e\n", x[0], x[1], fcost);

	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		steepest_descent_run(n, x, fcost, grad, &opt, &flag);
		if (flag == OPT_GRAD)
			rosenbrock(x, &fcost, grad);
	}

	printf("  Final:   x=(%.6f, %.6f), f=%.6e\n", x[0], x[1], fcost);
	printf("  Iterations: %d, Forward problems: %d\n",
	       opt.cpt_iter, opt.nfwd_pb);

	if (flag == OPT_FAIL) {
		printf("  FAIL: linesearch failed\n");
		return 1;
	}
	/* SD may not get very close with limited iterations, check loosely */
	if (fcost > 1e-3f) {
		printf("  FAIL: cost too high (%.6e)\n", fcost);
		return 1;
	}

	printf("  PASS\n\n");
	return 0;
}


/*--------------------------------------------------------------------
 * test_lbfgs_bounds -- Test L-BFGS with box constraints.
 *--------------------------------------------------------------------*/
static int test_lbfgs_bounds(void)
{
	int n = 2;
	float x[2] = {1.5f, 1.5f};
	float fcost, grad[2];
	float lb[2] = {0.0f, 0.0f};
	float ub[2] = {2.0f, 2.0f};
	optim_type opt;
	optFlag flag;

	printf("=== TEST 3: L-BFGS with bounds [0,2]x[0,2] ===\n");

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 10000;
	opt.conv = 1e-8f;
	opt.print_flag = 0;  /* Don't overwrite iterate_LB.dat from test 1 */
	opt.debug = 0;
	opt.l = 20;
	opt.bound = 1;
	opt.threshold = 0.0f;
	opt.lb = lb;
	opt.ub = ub;

	rosenbrock(x, &fcost, grad);
	printf("  Initial: x=(%.4f, %.4f), f=%.6e\n", x[0], x[1], fcost);

	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		lbfgs_run(n, x, fcost, grad, &opt, &flag);
		if (flag == OPT_GRAD)
			rosenbrock(x, &fcost, grad);
	}

	printf("  Final:   x=(%.6f, %.6f), f=%.6e\n", x[0], x[1], fcost);
	printf("  Iterations: %d\n", opt.cpt_iter);

	if (flag == OPT_FAIL) {
		printf("  FAIL: linesearch failed\n");
		return 1;
	}
	/* Minimum (1,1) is inside bounds, so should converge normally */
	if (fabsf(x[0] - 1.0f) > TOL || fabsf(x[1] - 1.0f) > TOL) {
		printf("  FAIL: did not converge to (1,1)\n");
		return 1;
	}

	printf("  PASS\n\n");
	return 0;
}


/*--------------------------------------------------------------------
 * test_pnlcg -- Test PNLCG (Dai-Yuan CG) on Rosenbrock.
 *
 * No preconditioning: grad_preco = grad.
 *--------------------------------------------------------------------*/
static int test_pnlcg(void)
{
	int n = 2;
	float x[2] = {1.5f, 1.5f};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;

	printf("=== TEST 4: PNLCG (Dai-Yuan) on Rosenbrock ===\n");

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 50000;
	opt.conv = 1e-8f;
	opt.print_flag = 1;
	opt.debug = 0;

	rosenbrock(x, &fcost, grad);
	printf("  Initial: x=(%.4f, %.4f), f=%.6e\n", x[0], x[1], fcost);

	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		/* No preconditioning: pass grad as grad_preco */
		pnlcg_run(n, x, fcost, grad, grad, &opt, &flag);
		if (flag == OPT_GRAD)
			rosenbrock(x, &fcost, grad);
	}

	printf("  Final:   x=(%.6f, %.6f), f=%.6e\n", x[0], x[1], fcost);
	printf("  Iterations: %d, Forward problems: %d\n",
	       opt.cpt_iter, opt.nfwd_pb);

	if (flag == OPT_FAIL) {
		printf("  FAIL: linesearch failed\n");
		return 1;
	}
	if (fabsf(x[0] - 1.0f) > TOL || fabsf(x[1] - 1.0f) > TOL) {
		printf("  FAIL: did not converge to (1,1)\n");
		return 1;
	}

	printf("  PASS\n\n");
	return 0;
}


/*--------------------------------------------------------------------
 * test_plbfgs -- Test PLBFGS on Rosenbrock.
 *
 * No preconditioning: grad_preco = grad, preconditioner is identity
 * (q_plb is not modified between descent1 and descent2).
 *--------------------------------------------------------------------*/
static int test_plbfgs(void)
{
	int n = 2;
	float x[2] = {1.5f, 1.5f};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;

	printf("=== TEST 5: PLBFGS on Rosenbrock ===\n");

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 10000;
	opt.conv = 1e-8f;
	opt.print_flag = 1;
	opt.debug = 0;
	opt.l = 20;

	rosenbrock(x, &fcost, grad);
	printf("  Initial: x=(%.4f, %.4f), f=%.6e\n", x[0], x[1], fcost);

	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		/* No preconditioning: pass grad as grad_preco */
		plbfgs_run(n, x, fcost, grad, grad, &opt, &flag);
		if (flag == OPT_PREC) {
			/* Identity preconditioner: do nothing to q_plb */
			continue;
		}
		if (flag == OPT_GRAD)
			rosenbrock(x, &fcost, grad);
	}

	printf("  Final:   x=(%.6f, %.6f), f=%.6e\n", x[0], x[1], fcost);
	printf("  Iterations: %d, Forward problems: %d\n",
	       opt.cpt_iter, opt.nfwd_pb);

	if (flag == OPT_FAIL) {
		printf("  FAIL: linesearch failed\n");
		return 1;
	}
	if (fabsf(x[0] - 1.0f) > TOL || fabsf(x[1] - 1.0f) > TOL) {
		printf("  FAIL: did not converge to (1,1)\n");
		return 1;
	}

	printf("  PASS\n\n");
	return 0;
}


/*--------------------------------------------------------------------
 * test_trn -- Test TRN (Truncated Newton) on Rosenbrock.
 *
 * Uses exact Hessian-vector product for the inner CG solver.
 *--------------------------------------------------------------------*/
static int test_trn(void)
{
	int n = 2;
	float x[2] = {1.5f, 1.5f};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;

	printf("=== TEST 6: TRN (Truncated Newton) on Rosenbrock ===\n");

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 100;     /* Match Fortran test_TRN */
	opt.conv = 1e-8f;
	opt.print_flag = 1;
	opt.debug = 0;
	opt.niter_max_CG = 5;   /* Match Fortran test_TRN */

	rosenbrock(x, &fcost, grad);
	printf("  Initial: x=(%.4f, %.4f), f=%.6e\n", x[0], x[1], fcost);

	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		trn_run(n, x, fcost, grad, &opt, &flag);
		if (flag == OPT_HESS) {
			/* Compute Hd = H * d at current x */
			rosenbrock_hessian_vec(x, opt.d, opt.Hd);
		} else if (flag == OPT_GRAD) {
			rosenbrock(x, &fcost, grad);
		} else if (flag == OPT_NSTE) {
			/* New step: recompute cost/gradient for next CG */
			rosenbrock(x, &fcost, grad);
		}
	}

	printf("  Final:   x=(%.6f, %.6f), f=%.6e\n", x[0], x[1], fcost);
	printf("  Iterations: %d, Forward problems: %d, Hess-vec: %d\n",
	       opt.cpt_iter, opt.nfwd_pb, opt.nhess);

	if (flag == OPT_FAIL) {
		printf("  FAIL: linesearch failed\n");
		return 1;
	}
	if (fabsf(x[0] - 1.0f) > TOL || fabsf(x[1] - 1.0f) > TOL) {
		printf("  FAIL: did not converge to (1,1)\n");
		return 1;
	}

	printf("  PASS\n\n");
	return 0;
}


int main(void)
{
	int nfail = 0;

	printf("Optimization Library Unit Tests\n");
	printf("================================\n\n");

	nfail += test_lbfgs();
	nfail += test_sd();
	nfail += test_lbfgs_bounds();
	nfail += test_pnlcg();
	nfail += test_plbfgs();
	nfail += test_trn();

	printf("================================\n");
	if (nfail == 0)
		printf("All tests PASSED\n");
	else
		printf("%d test(s) FAILED\n", nfail);

	return nfail;
}
