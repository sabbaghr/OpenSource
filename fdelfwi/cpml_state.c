#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"

/**
 * cpml_state.c  --  Ownership of the FORWARD CFS-CPML psi memory variables.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The CFS-CPML recursion
 *
 *     psi^{n+1} = b * psi^n + c * D(u^n)
 *
 * carries state.  psi^{n+1} depends on the ENTIRE history of the wavefield
 * inside the absorbing band, not just on u^n.  So {vx, vz, txx, tzz, txz} is
 * NOT a complete description of the forward simulation when a PML is active:
 * the full state is {vx, vz, txx, tzz, txz, psi}.
 *
 * adj_shot/born_shot/precond_shot reconstruct the forward wavefield by
 * restarting it from a checkpoint.  If psi is not restored along with the
 * wavefield, each segment restarts with psi = 0 and the reconstructed forward
 * field is wrong in and near the PML -- which corrupts the gradient wherever
 * the PML overlaps anything that matters (a source or receiver a few cells
 * from the boundary, for instance).
 *
 * psi used to live in function-static arrays inside boundariesP/boundariesV,
 * allocated on the first time step and FREED on the last, which made it both
 * unreachable from checkpoint.c and dead by the time the adjoint ran.  This
 * file owns the arrays instead, with an explicit lifetime, so that
 * checkpoint.c can save and restore them exactly.
 *
 * LAYOUT.  The serialised buffer is CPML_NPSI arrays of n = nax*naz floats,
 * concatenated in the fixed order of cpmlStatePtrs() below.  The order is
 * fixed so a checkpoint file written by one run is readable by the next.
 *
 * Note that psi is identically zero outside the absorbing band, so most of
 * the buffer is zeros.  It is stored in full anyway: the band index ranges
 * differ per component and per active edge, and duplicating that bookkeeping
 * here is a far better way to introduce a subtle bug than it is to save disk.
 */

/*
 * CONTEXTS.  psi belongs to a WAVEFIELD, not to the grid.  born_shot runs the
 * background re-propagation and the Born scattered field in the same time
 * loop, both through boundariesP/boundariesV, so a single psi set would make
 * each field's PML memory depend on the other's history.  That is not a
 * stale-buffer nuisance; it silently changes the operator, and it broke the
 * Gauss-Newton Hessian symmetry test (relative error 2.2e-01 under CPML x4
 * and 2.0e+00 with a free surface, against 4.5e-07 under taper).
 *
 * Contexts are independent psi sets.  Callers that run one wavefield never
 * touch this and stay on CPML_CTX_FWD.
 */
static float *g_psi[CPML_NCTX][CPML_NPSI];
static size_t g_n[CPML_NCTX];
static int    g_ctx = CPML_CTX_FWD;


/***********************************************************************
 * cpmlStatePtrs -- The fixed serialisation order, for the ACTIVE context.
 ***********************************************************************/
static float **cpmlStatePtrs(void)
{
	return g_psi[g_ctx];
}


/***********************************************************************
 * cpmlStateSelect -- Make ctx the active psi set.
 *
 * Cheap and allocation-free: the arrays are created on demand by the next
 * cpmlStateAlloc, which boundariesP/boundariesV call on every time step.
 ***********************************************************************/
void cpmlStateSelect(int ctx)
{
	if (ctx < 0 || ctx >= CPML_NCTX)
		verr("cpmlStateSelect: context %d out of range [0,%d)", ctx, CPML_NCTX);
	g_ctx = ctx;
}


/***********************************************************************
 * cpmlStateContext -- The active context, so a caller can save/restore it
 * instead of assuming it owns the selection.
 ***********************************************************************/
int cpmlStateContext(void)
{
	return g_ctx;
}


/***********************************************************************
 * cpmlStateAlloc -- Allocate (zeroed) the psi arrays for an n-point grid.
 *
 * Idempotent: a second call with the same n returns the same buffers with
 * their contents intact, so repeated shots reuse one allocation.  A call
 * with a different n discards and reallocates.
 *
 * NOT thread safe -- call it from a single thread (the existing boundariesP /
 * boundariesV allocation blocks already run under "#pragma omp master").
 *
 * Returns the base of the array table; index it with the CPML_PSI_* enum.
 ***********************************************************************/
float **cpmlStateAlloc(size_t n)
{
	int i;

	if (n == 0) return NULL;

	if (g_n[g_ctx] == n && g_psi[g_ctx][0] != NULL) return cpmlStatePtrs();

	cpmlStateFree();

	for (i = 0; i < CPML_NPSI; i++) {
		g_psi[g_ctx][i] = (float *)calloc(n, sizeof(float));
		if (!g_psi[g_ctx][i]) {
			verr("cpmlStateAlloc: cannot allocate psi array %d (%zu floats)", i, n);
			return NULL;
		}
	}
	g_n[g_ctx] = n;

	return cpmlStatePtrs();
}


/***********************************************************************
 * cpmlStateFree -- Release the psi arrays.
 ***********************************************************************/
void cpmlStateFree(void)
{
	int i;

	for (i = 0; i < CPML_NPSI; i++) {
		if (g_psi[g_ctx][i]) { free(g_psi[g_ctx][i]); g_psi[g_ctx][i] = NULL; }
	}
	g_n[g_ctx] = 0;
}


/***********************************************************************
 * cpmlStateFloats -- Size of the serialised state, in floats.
 *
 * Zero when no PML has ever been allocated, which is how checkpoint.c
 * decides whether a psi checkpoint file is needed at all.
 ***********************************************************************/
size_t cpmlStateFloats(void)
{
	return (g_psi[g_ctx][0] != NULL) ? (size_t)CPML_NPSI * g_n[g_ctx] : 0;
}


/***********************************************************************
 * cpmlStateSave -- Serialise psi into buf (CPML_NPSI * n floats).
 *
 * Returns 0 on success, -1 if there is no state or n disagrees.
 ***********************************************************************/
int cpmlStateSave(float *buf, size_t n)
{
	int i;

	if (!buf || g_psi[g_ctx][0] == NULL) return -1;
	if (n != g_n[g_ctx]) {
		verr("cpmlStateSave: grid size mismatch (have %zu, asked %zu)", g_n[g_ctx], n);
		return -1;
	}

	for (i = 0; i < CPML_NPSI; i++)
		memcpy(buf + (size_t)i * n, g_psi[g_ctx][i], n * sizeof(float));

	return 0;
}


/***********************************************************************
 * cpmlStateLoad -- Restore psi from buf (CPML_NPSI * n floats).
 *
 * Allocates the arrays if they do not exist yet, so a checkpoint can be
 * restored before the first FD time step of the reconstruction has run.
 *
 * Returns 0 on success, -1 on failure.
 ***********************************************************************/
int cpmlStateLoad(const float *buf, size_t n)
{
	int i;

	if (!buf || n == 0) return -1;
	if (g_psi[g_ctx][0] == NULL || g_n[g_ctx] != n) {
		if (!cpmlStateAlloc(n)) return -1;
	}

	for (i = 0; i < CPML_NPSI; i++)
		memcpy(g_psi[g_ctx][i], buf + (size_t)i * n, n * sizeof(float));

	return 0;
}


/***********************************************************************
 * cpmlStateZero -- Zero psi without deallocating.
 *
 * This is the per-shot reset: boundariesP/boundariesV call it at itime==0,
 * which is exactly the initial condition psi(t=0) = 0.
 ***********************************************************************/
void cpmlStateZero(void)
{
	int i;

	if (g_psi[g_ctx][0] == NULL) return;
	for (i = 0; i < CPML_NPSI; i++)
		memset(g_psi[g_ctx][i], 0, g_n[g_ctx] * sizeof(float));
}
