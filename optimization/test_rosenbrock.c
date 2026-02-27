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


/*--------------------------------------------------------------------
 * test_enriched -- Test Enriched (L-BFGS + TN hybrid) on Rosenbrock.
 *
 * Starts with TRN to seed L-BFGS buffer, then switches to L-BFGS.
 *--------------------------------------------------------------------*/
static int test_enriched(void)
{
	int n = 2;
	float x[2] = {1.5f, 1.5f};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;

	printf("=== TEST 7: Enriched (L-BFGS + TN) on Rosenbrock ===\n");

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 100;
	opt.conv = 1e-8f;
	opt.print_flag = 1;
	opt.debug = 0;
	opt.l = 20;           /* L-BFGS memory */
	opt.enr_l = 20;       /* L-BFGS cycle length */
	opt.enr_maxcg = 5;    /* max CG iterations */

	rosenbrock(x, &fcost, grad);
	printf("  Initial: x=(%.4f, %.4f), f=%.6e\n", x[0], x[1], fcost);

	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		enriched_run(n, x, fcost, grad, &opt, &flag);
		if (flag == OPT_HESS) {
			rosenbrock_hessian_vec(x, opt.d, opt.Hd);
		} else if (flag == OPT_GRAD) {
			rosenbrock(x, &fcost, grad);
		} else if (flag == OPT_NSTE) {
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


/*--------------------------------------------------------------------
 * Comparison benchmark infrastructure.
 *--------------------------------------------------------------------*/
typedef struct {
	const char *name;
	int converged;
	int iters;
	int ngrad;
	int nhess;
	float final_cost;
	float final_x[2];
} bench_result;

static bench_result run_bench_lbfgs(float x0, float x1)
{
	int n = 2;
	float x[2] = {x0, x1};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;
	bench_result r;

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 10000;
	opt.conv = 1e-8f;
	opt.print_flag = 0;
	opt.l = 20;

	rosenbrock(x, &fcost, grad);
	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		lbfgs_run(n, x, fcost, grad, &opt, &flag);
		if (flag == OPT_GRAD) rosenbrock(x, &fcost, grad);
	}

	r.name = "L-BFGS";
	r.converged = (flag == OPT_CONV);
	r.iters = opt.cpt_iter;
	r.ngrad = opt.nfwd_pb;
	r.nhess = 0;
	r.final_cost = fcost;
	r.final_x[0] = x[0]; r.final_x[1] = x[1];
	return r;
}

static bench_result run_bench_pnlcg(float x0, float x1)
{
	int n = 2;
	float x[2] = {x0, x1};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;
	bench_result r;

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 50000;
	opt.conv = 1e-8f;
	opt.print_flag = 0;

	rosenbrock(x, &fcost, grad);
	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		pnlcg_run(n, x, fcost, grad, grad, &opt, &flag);
		if (flag == OPT_GRAD) rosenbrock(x, &fcost, grad);
	}

	r.name = "PNLCG";
	r.converged = (flag == OPT_CONV);
	r.iters = opt.cpt_iter;
	r.ngrad = opt.nfwd_pb;
	r.nhess = 0;
	r.final_cost = fcost;
	r.final_x[0] = x[0]; r.final_x[1] = x[1];
	return r;
}

static bench_result run_bench_trn(float x0, float x1)
{
	int n = 2;
	float x[2] = {x0, x1};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;
	bench_result r;

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 10000;
	opt.conv = 1e-8f;
	opt.print_flag = 0;
	opt.niter_max_CG = 5;

	rosenbrock(x, &fcost, grad);
	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		trn_run(n, x, fcost, grad, &opt, &flag);
		if (flag == OPT_HESS) rosenbrock_hessian_vec(x, opt.d, opt.Hd);
		else if (flag == OPT_GRAD || flag == OPT_NSTE) rosenbrock(x, &fcost, grad);
	}

	r.name = "TRN";
	r.converged = (flag == OPT_CONV);
	r.iters = opt.cpt_iter;
	r.ngrad = opt.nfwd_pb;
	r.nhess = opt.nhess;
	r.final_cost = fcost;
	r.final_x[0] = x[0]; r.final_x[1] = x[1];
	return r;
}

static bench_result run_bench_enriched(float x0, float x1)
{
	int n = 2;
	float x[2] = {x0, x1};
	float fcost, grad[2];
	optim_type opt;
	optFlag flag;
	bench_result r;

	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = 10000;
	opt.conv = 1e-8f;
	opt.print_flag = 0;
	opt.l = 20;
	opt.enr_l = 20;
	opt.enr_maxcg = 5;

	rosenbrock(x, &fcost, grad);
	flag = OPT_INIT;
	while (flag != OPT_CONV && flag != OPT_FAIL) {
		enriched_run(n, x, fcost, grad, &opt, &flag);
		if (flag == OPT_HESS) rosenbrock_hessian_vec(x, opt.d, opt.Hd);
		else if (flag == OPT_GRAD || flag == OPT_NSTE) rosenbrock(x, &fcost, grad);
	}

	r.name = "Enriched";
	r.converged = (flag == OPT_CONV);
	r.iters = opt.cpt_iter;
	r.ngrad = opt.nfwd_pb;
	r.nhess = opt.nhess;
	r.final_cost = fcost;
	r.final_x[0] = x[0]; r.final_x[1] = x[1];
	return r;
}

static void run_comparison(float x0, float x1)
{
	float fcost, grad[2], x[2] = {x0, x1};
	rosenbrock(x, &fcost, grad);

	printf("================================================================\n");
	printf("  Rosenbrock f(x)=(1-x)^2+100(y-x^2)^2\n");
	printf("  Start: (%.1f, %.1f)  f0=%.2e  conv=1e-8\n", x0, x1, fcost);
	printf("================================================================\n");
	printf("  %-12s %6s %6s %6s %10s %10s\n",
	       "Method", "Iter", "Grads", "Hess", "Final cost", "Status");
	printf("  %-12s %6s %6s %6s %10s %10s\n",
	       "------", "----", "-----", "----", "----------", "------");

	bench_result results[4];
	results[0] = run_bench_pnlcg(x0, x1);
	results[1] = run_bench_lbfgs(x0, x1);
	results[2] = run_bench_trn(x0, x1);
	results[3] = run_bench_enriched(x0, x1);

	for (int i = 0; i < 4; i++) {
		printf("  %-12s %6d %6d %6d %10.2e %10s\n",
		       results[i].name,
		       results[i].iters,
		       results[i].ngrad,
		       results[i].nhess,
		       results[i].final_cost,
		       results[i].converged ? "CONV" : "FAIL");
	}
	printf("================================================================\n");
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
	nfail += test_enriched();

	printf("================================\n");
	if (nfail == 0)
		printf("All tests PASSED\n");
	else
		printf("%d test(s) FAILED\n", nfail);

	/* Comparison benchmarks from multiple starting points */
	printf("\n\n");
	run_comparison(-1.2f, 1.0f);
	printf("\n");
	run_comparison(-5.0f, 5.0f);

	return nfail;
}
