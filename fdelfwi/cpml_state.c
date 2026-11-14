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

static float *g_psi[CPML_NPSI];
static size_t g_n = 0;


/***********************************************************************
 * cpmlStatePtrs -- The fixed serialisation order.
 ***********************************************************************/
static float **cpmlStatePtrs(void)
{
	return g_psi;
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

	if (g_n == n && g_psi[0] != NULL) return cpmlStatePtrs();

	cpmlStateFree();

	for (i = 0; i < CPML_NPSI; i++) {
		g_psi[i] = (float *)calloc(n, sizeof(float));
		if (!g_psi[i]) {
			verr("cpmlStateAlloc: cannot allocate psi array %d (%zu floats)", i, n);
			return NULL;
		}
	}
	g_n = n;

	return cpmlStatePtrs();
}


/***********************************************************************
 * cpmlStateFree -- Release the psi arrays.
 ***********************************************************************/
void cpmlStateFree(void)
{
	int i;

	for (i = 0; i < CPML_NPSI; i++) {
		if (g_psi[i]) { free(g_psi[i]); g_psi[i] = NULL; }
	}
	g_n = 0;
}


/***********************************************************************
 * cpmlStateFloats -- Size of the serialised state, in floats.
 *
 * Zero when no PML has ever been allocated, which is how checkpoint.c
 * decides whether a psi checkpoint file is needed at all.
 ***********************************************************************/
size_t cpmlStateFloats(void)
{
	return (g_psi[0] != NULL) ? (size_t)CPML_NPSI * g_n : 0;
}


/***********************************************************************
 * cpmlStateSave -- Serialise psi into buf (CPML_NPSI * n floats).
 *
 * Returns 0 on success, -1 if there is no state or n disagrees.
 ***********************************************************************/
int cpmlStateSave(float *buf, size_t n)
{
	int i;

	if (!buf || g_psi[0] == NULL) return -1;
	if (n != g_n) {
		verr("cpmlStateSave: grid size mismatch (have %zu, asked %zu)", g_n, n);
		return -1;
	}

	for (i = 0; i < CPML_NPSI; i++)
		memcpy(buf + (size_t)i * n, g_psi[i], n * sizeof(float));

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
	if (g_psi[0] == NULL || g_n != n) {
		if (!cpmlStateAlloc(n)) return -1;
	}

	for (i = 0; i < CPML_NPSI; i++)
		memcpy(g_psi[i], buf + (size_t)i * n, n * sizeof(float));

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

	if (g_psi[0] == NULL) return;
	for (i = 0; i < CPML_NPSI; i++)
		memset(g_psi[i], 0, g_n * sizeof(float));
}
