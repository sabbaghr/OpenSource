#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"fdelfwi.h"
#include"segy.h"
#include"par.h"

/**
 * readResidual: Reads residual (adjoint source) traces from a Seismic Unix
 * (SU) file for one shot gather. Receiver type and orientation are determined
 * from the TRID header following the fdacrtmc/readRcvWav.c convention:
 *
 *   typ    = (trid - 1) % 8 + 1
 *   orient = (trid - 1) / 8 + 1
 *
 *   Monopole (orient=1):        1=P  2=Txz  3=Tzz  4=Txx  5=S-pot  6=Fx  7=Fz  8=P-pot
 *   Vertical Dipole (orient=2): 9=P 10=Txz 11=Tzz 12=Txx 13=S-pot 14=Fx 15=Fz 16=P-pot
 *   Horizontal Dipole (orient=3): 17-24 same pattern
 *
 * Receiver locations are mapped to the staggered grid:
 *   Types 2,6 (Txz, Fx) -> Vx grid (ioXx)
 *   Types 2,7 (Txz, Fz) -> Vz grid (ioZz) for vertical
 *   All others           -> P/Txx/Tzz grid (ioPx, ioPz)
 *
 * AUTHOR:
 *   Based on fdacrtmc/readRcvWav.c by Max Holicki.
 *   Adapted for fdelfwi data structures.
 */

int readResidual(const char *filename, adjSrcPar *adj, modPar *mod, bndPar *bnd)
{
	FILE *fp;
	segy hdr;
	size_t count, isrc;
	float scalco, scalel;
	float xmax, zmax;
	int fldr_match;
	int ibndx, ibndz;

	if (!filename || !adj || !mod || !bnd) {
		verr("readResidual: NULL pointer argument.");
		return -1;
	}

	xmax = mod->x0 + mod->dx * mod->nx;
	zmax = mod->z0 + mod->dz * mod->nz;

	/* Compute boundary offsets for P/Txx/Tzz grid positions,
	 * matching the convention in getRecTimes.c.
	 * ioPx/ioPz do NOT include bnd.ntap; ioXx/ioZz already do. */
	ibndx = mod->ioPx;
	ibndz = mod->ioPz;
	if (bnd->lef == 4 || bnd->lef == 2) ibndx += bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibndz += bnd->ntap;

	/*************/
	/* Open File */
	/*************/
	fp = fopen(filename, "r");
	if (!fp) verr("readResidual: Unable to open residual file: %s", filename);

	/*******************************************/
	/* Read First Trace Header                 */
	/*******************************************/
	count = fread(&hdr, 1, TRCBYTES, fp);
	if (count != TRCBYTES) {
		if (feof(fp)) {
			if (!count) verr("readResidual: Residual file %s is empty.", filename);
			else verr("readResidual: EOF while reading header of trace 1 in %s.", filename);
		} else {
			verr("readResidual: Could not read header of trace 1 in %s.", filename);
		}
	}

	fldr_match = hdr.fldr;

	if (hdr.trid < 1 || hdr.trid > 24)
		verr("readResidual: In %s trace 1 has unknown receiver type (trid=%d).", filename, hdr.trid);
	if (hdr.ns < 1)
		verr("readResidual: In %s trace 1 has invalid ns=%d.", filename, hdr.ns);

	/********************************************/
	/* Count Traces in This Shot Gather         */
	/********************************************/
	isrc = 0;
	do {
		if (fseeko(fp, (off_t)hdr.ns * sizeof(float), SEEK_CUR))
			perror("readResidual: fseek");
		count = fread(&hdr, 1, TRCBYTES, fp);
		if (count != TRCBYTES) {
			if (feof(fp)) {
				if (!count) { isrc++; break; }
				else verr("readResidual: EOF while reading header of trace %zu in %s.", isrc + 2, filename);
			} else {
				verr("readResidual: Could not read header of trace %zu in %s.", isrc + 2, filename);
			}
		}
		isrc++;
	} while (hdr.fldr == fldr_match);

	/* Seek back to start of file */
	if (fseeko(fp, 0, SEEK_SET))
		perror("readResidual: fseek to start");

	/******************/
	/* Allocate Arrays */
	/******************/
	adj->nsrc = (int)isrc;
	adj->xi     = (size_t *)malloc(adj->nsrc * sizeof(size_t));
	adj->zi     = (size_t *)malloc(adj->nsrc * sizeof(size_t));
	adj->typ    = (int *)malloc(adj->nsrc * sizeof(int));
	adj->orient = (int *)malloc(adj->nsrc * sizeof(int));
	adj->x      = (float *)malloc(adj->nsrc * sizeof(float));
	adj->z      = (float *)malloc(adj->nsrc * sizeof(float));

	if (!adj->xi || !adj->zi || !adj->typ || !adj->orient || !adj->x || !adj->z)
		verr("readResidual: Memory allocation failed for %d traces.", adj->nsrc);

	/*************/
	/* Read Data */
	/*************/
	for (isrc = 0; isrc < (size_t)adj->nsrc; isrc++) {
		float grid_x, grid_z;

		if (fread(&hdr, 1, TRCBYTES, fp) != TRCBYTES)
			verr("readResidual: Could not read header for trace %zu in %s.", isrc + 1, filename);

		if (hdr.trid < 1 || hdr.trid > 24)
			verr("readResidual: In %s trace %zu has unknown receiver type (trid=%d).", filename, isrc + 1, hdr.trid);

		/* Set nt from first trace */
		if (isrc == 0) {
			adj->nt  = (int)hdr.ns;
			adj->wav = (float *)malloc((size_t)adj->nsrc * adj->nt * sizeof(float));
			if (!adj->wav)
				verr("readResidual: Memory allocation failed for waveform data (%d x %d).", adj->nsrc, adj->nt);
		} else {
			if ((int)hdr.ns != adj->nt)
				verr("readResidual: In %s trace %zu has %d samples, expected %d.", filename, isrc + 1, hdr.ns, adj->nt);
		}

		/* Type and orientation from TRID */
		adj->typ[isrc]    = (hdr.trid - 1) % 8 + 1;
		adj->orient[isrc] = (hdr.trid - 1) / 8 + 1;

		/* Horizontal receiver location */
		if (hdr.scalco < 0)       scalco = 1.0f / (-(float)hdr.scalco);
		else if (hdr.scalco > 0)  scalco = (float)hdr.scalco;
		else                      scalco = 1.0f;

		grid_x = (((float)hdr.gx) * scalco - mod->x0) / mod->dx + 0.5f;
		if (grid_x < 0 || grid_x > mod->nx)
			verr("readResidual: In %s trace %zu lies outside model bounds (x). gx=%d, x0=%.1f, xmax=%.1f",
				filename, isrc + 1, hdr.gx, mod->x0, xmax);

		adj->x[isrc] = truncf(grid_x) * mod->dx + mod->x0;

		/* Map to staggered grid: types 2,6 (Txz,Fx) use Vx grid
		 * (ioXx already includes ntap from getParameters).
		 * All others use P/Txx/Tzz grid (ibndx = ioPx + ntap). */
		if (adj->typ[isrc] == 2 || adj->typ[isrc] == 6)
			adj->xi[isrc] = mod->ioXx + (size_t)grid_x;
		else
			adj->xi[isrc] = ibndx + (size_t)grid_x;

		/* Vertical receiver location */
		if (hdr.scalel < 0)       scalel = 1.0f / (-(float)hdr.scalel);
		else if (hdr.scalel > 0)  scalel = (float)hdr.scalel;
		else                      scalel = 1.0f;

		grid_z = (((float)(-hdr.gelev)) * scalel - mod->z0) / mod->dz + 0.5f;
		if (grid_z < 0 || grid_z > mod->nz)
			verr("readResidual: In %s trace %zu lies outside model bounds (z). gelev=%d, z0=%.1f, zmax=%.1f",
				filename, isrc + 1, hdr.gelev, mod->z0, zmax);

		adj->z[isrc] = truncf(grid_z) * mod->dz + mod->z0;

		/* Map to staggered grid: types 2,7 (Txz,Fz) use Vz grid
		 * (ioZz already includes ntap from getParameters).
		 * All others use P/Txx/Tzz grid (ibndz = ioPz + ntap).
		 *
		 * For type 7 (Fz/Vz): getRecTimes records vz at iz2=iz+1
		 * where iz = ibndz + grid_z.  Since ioZz = ioPz + 1 + ntap
		 * while ibndz = ioPz + ntap, we have ioZz = ibndz + 1.
		 * The stagger +1 is already in ioZz, so NO extra +1 needed:
		 * ioZz + grid_z = (ibndz+1) + grid_z = ibndz + grid_z + 1 = iz2. */
		if (adj->typ[isrc] == 7)
			adj->zi[isrc] = mod->ioZz + (size_t)grid_z;
		else if (adj->typ[isrc] == 2)
			adj->zi[isrc] = mod->ioZz + (size_t)grid_z;
		else
			adj->zi[isrc] = ibndz + (size_t)grid_z;

		/* Read trace data */
		if (fread(&adj->wav[isrc * adj->nt], sizeof(float), adj->nt, fp) != (size_t)adj->nt)
			verr("readResidual: Could not read data for trace %zu in %s.", isrc + 1, filename);
	}

	fclose(fp);
	return 0;
}


void freeResidual(adjSrcPar *adj)
{
	if (!adj) return;
	if (adj->xi)     { free(adj->xi);     adj->xi     = NULL; }
	if (adj->zi)     { free(adj->zi);     adj->zi     = NULL; }
	if (adj->typ)    { free(adj->typ);    adj->typ    = NULL; }
	if (adj->orient) { free(adj->orient); adj->orient = NULL; }
	if (adj->x)      { free(adj->x);      adj->x      = NULL; }
	if (adj->z)      { free(adj->z);      adj->z      = NULL; }
	if (adj->wav)    { free(adj->wav);    adj->wav    = NULL; }
	adj->nsrc = 0;
	adj->nt   = 0;
}
