#include<stdlib.h>
#include<stdio.h>
#include"par.h"
#include"segy.h"
#include"fdelfwi.h"

/**
 * computeHydrophone.c -- Elastic hydrophone (pressure) from the normal
 *                        stresses, P = 0.5*(Txx + Tzz).
 *
 * fdelmodc's rec_type_p=1 does NOT give a usable elastic hydrophone, so the
 * pressure channel has to be synthesised from recorded Txx and Tzz.  Every
 * place that models data and then wants a `_rp.su` has to do this: the
 * forward/synthetic path, the observed-data path, and the Born path.
 *
 * This file exists because that function was copy-pasted into four drivers
 * (fwi_inversion.c, fwi_driver.c, hess_shot.c, test_gradient_visual.c as
 * file-local statics) and the copies drifted: Born modelling writes _rtxx and
 * _rtzz but never built _rp, so comp=_rp silently used a stale or missing
 * pressure file until a copy was added to hess_shot.c.  New code should call
 * THIS one rather than add a fifth static.
 *
 * The output trace id is set to 11 (pressure) so downstream readResidual /
 * applyAdjointSource map it to the hydrophone adjoint injection.
 *
 * Returns 0 on success, -1 if either input cannot be opened or read, or if
 * the output cannot be created.  Trace pairs are consumed in file order and
 * the loop stops at the first short read or `ns` mismatch, so a truncated
 * input yields a correspondingly short output rather than garbage.
 */
int computeHydrophone(const char *file_tzz, const char *file_txx,
                      const char *file_p, int verbose)
{
	FILE *fp_tzz, *fp_txx, *fp_p;
	segy hdr_tzz, hdr_txx;
	float *data_tzz, *data_txx;
	int ns, is, ntrace = 0;
	size_t nread;

	fp_tzz = fopen(file_tzz, "r");
	fp_txx = fopen(file_txx, "r");
	if (!fp_tzz || !fp_txx) {
		if (fp_tzz) fclose(fp_tzz);
		if (fp_txx) fclose(fp_txx);
		return -1;
	}

	fp_p = fopen(file_p, "w");
	if (!fp_p) { fclose(fp_tzz); fclose(fp_txx); return -1; }

	nread = fread(&hdr_tzz, 1, TRCBYTES, fp_tzz);
	if (nread != TRCBYTES) {
		fclose(fp_tzz); fclose(fp_txx); fclose(fp_p);
		return -1;
	}
	rewind(fp_tzz);

	ns = hdr_tzz.ns;
	data_tzz = (float *)malloc(ns * sizeof(float));
	data_txx = (float *)malloc(ns * sizeof(float));
	if (!data_tzz || !data_txx) {
		free(data_tzz); free(data_txx);
		fclose(fp_tzz); fclose(fp_txx); fclose(fp_p);
		return -1;
	}

	while (fread(&hdr_tzz, 1, TRCBYTES, fp_tzz) == TRCBYTES &&
	       fread(&hdr_txx, 1, TRCBYTES, fp_txx) == TRCBYTES) {
		if (hdr_tzz.ns != ns || hdr_txx.ns != ns) break;
		nread = fread(data_tzz, sizeof(float), ns, fp_tzz);
		if ((int)nread != ns) break;
		nread = fread(data_txx, sizeof(float), ns, fp_txx);
		if ((int)nread != ns) break;
		for (is = 0; is < ns; is++)
			data_tzz[is] = 0.5f * (data_tzz[is] + data_txx[is]);
		hdr_tzz.trid = 11;              /* pressure */
		fwrite(&hdr_tzz, 1, TRCBYTES, fp_p);
		fwrite(data_tzz, sizeof(float), ns, fp_p);
		ntrace++;
	}

	free(data_tzz); free(data_txx);
	fclose(fp_tzz); fclose(fp_txx); fclose(fp_p);

	if (verbose > 1)
		vmess("computeHydrophone: %s = 0.5*(%s + %s), %d traces, ns=%d",
			file_p, file_txx, file_tzz, ntrace, ns);

	return 0;
}
