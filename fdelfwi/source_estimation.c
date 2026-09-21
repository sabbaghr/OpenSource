#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "segy.h"

/*--------------------------------------------------------------------
 * source_estimation.c -- cc1 source amplitude estimation.
 *
 * Matches TOY2DAC V2.6 (sub_FDFD.f90, lines 717-734).
 *
 * In the time domain, cc1 is a real scalar:
 *   cc1 = sum_k sum_t d_cal[k,t] * d_obs[k,t]
 *        / sum_k sum_t d_cal[k,t] * d_cal[k,t]
 *
 * This is the least-squares optimal amplitude scaling:
 *   min_cc1 ||d_obs - cc1 * d_cal||^2
 *   => cc1 = <d_cal, d_obs> / <d_cal, d_cal>
 *
 * Options:
 *   src_estim=0 : no estimation (cc1 = 1)
 *   src_estim=1 : one global cc1 for all sources
 *   src_estim=2 : one cc1 per source gather
 *
 * Computed at the first gradient evaluation only, then fixed.
 *
 * Reference: Pratt (1999), Geophysics, source wavelet estimation.
 *--------------------------------------------------------------------*/


/*--------------------------------------------------------------------
 * compute_cc1_shot -- Compute cc1 for a single shot gather.
 *
 * Reads synthetic and observed SU files and computes:
 *   cc1 = <d_cal, d_obs> / <d_cal, d_cal>
 *
 * Parameters:
 *   syn_file - path to synthetic data (.su)
 *   obs_file - path to observed data (.su)
 *   verbose  - print diagnostic info
 *
 * Returns: cc1 (real scalar, typically close to 1.0)
 *--------------------------------------------------------------------*/
float compute_cc1_shot(const char *syn_file, const char *obs_file,
                       int verbose)
{
    FILE *fp_syn, *fp_obs;
    segy hdr_syn, hdr_obs;
    float *buf_syn = NULL, *buf_obs = NULL;
    double num = 0.0, den = 0.0;
    int ns, ntr = 0;

    fp_syn = fopen(syn_file, "rb");
    fp_obs = fopen(obs_file, "rb");
    if (!fp_syn || !fp_obs) {
        if (fp_syn) fclose(fp_syn);
        if (fp_obs) fclose(fp_obs);
        fprintf(stderr, "source_estimation: cannot open %s or %s\n",
                syn_file, obs_file);
        return 1.0f;
    }

    while (fread(&hdr_syn, sizeof(segy), 1, fp_syn) == 1 &&
           fread(&hdr_obs, sizeof(segy), 1, fp_obs) == 1) {

        ns = (int)hdr_syn.ns;
        if (ntr == 0) {
            buf_syn = (float *)malloc(ns * sizeof(float));
            buf_obs = (float *)malloc(ns * sizeof(float));
        }

        if (fread(buf_syn, sizeof(float), ns, fp_syn) != (size_t)ns ||
            fread(buf_obs, sizeof(float), ns, fp_obs) != (size_t)ns)
            break;

        /* Accumulate dot products */
        for (int it = 0; it < ns; it++) {
            num += (double)buf_syn[it] * (double)buf_obs[it];
            den += (double)buf_syn[it] * (double)buf_syn[it];
        }
        ntr++;
    }

    fclose(fp_syn);
    fclose(fp_obs);
    free(buf_syn);
    free(buf_obs);

    if (den < 1.0e-30) {
        if (verbose)
            fprintf(stderr, "source_estimation: WARNING zero synthetic energy "
                    "for %s, cc1=1\n", syn_file);
        return 1.0f;
    }

    float cc1 = (float)(num / den);

    if (verbose)
        fprintf(stderr, "source_estimation: %s  cc1=%.6f  "
                "<cal,obs>=%.4e  <cal,cal>=%.4e  ntr=%d\n",
                syn_file, cc1, num, den, ntr);

    return cc1;
}


/*--------------------------------------------------------------------
 * compute_cc1_global -- Compute one global cc1 for all shots.
 *
 * Accumulates <d_cal, d_obs> and <d_cal, d_cal> across all shots
 * before dividing.
 *
 * Parameters:
 *   syn_files - array of synthetic file paths [nshots]
 *   obs_files - array of observed file paths [nshots]
 *   nshots    - number of shots
 *   verbose   - print diagnostic info
 *
 * Returns: global cc1
 *--------------------------------------------------------------------*/
float compute_cc1_global(const char **syn_files, const char **obs_files,
                         int nshots, int verbose)
{
    double num_total = 0.0, den_total = 0.0;
    int ishot;

    for (ishot = 0; ishot < nshots; ishot++) {
        FILE *fp_syn, *fp_obs;
        segy hdr;
        float *buf_syn = NULL, *buf_obs = NULL;
        int ns;

        fp_syn = fopen(syn_files[ishot], "rb");
        fp_obs = fopen(obs_files[ishot], "rb");
        if (!fp_syn || !fp_obs) {
            if (fp_syn) fclose(fp_syn);
            if (fp_obs) fclose(fp_obs);
            continue;
        }

        while (fread(&hdr, sizeof(segy), 1, fp_syn) == 1) {
            segy hdr_obs;
            if (fread(&hdr_obs, sizeof(segy), 1, fp_obs) != 1) break;

            ns = (int)hdr.ns;
            buf_syn = (float *)realloc(buf_syn, ns * sizeof(float));
            buf_obs = (float *)realloc(buf_obs, ns * sizeof(float));

            if (fread(buf_syn, sizeof(float), ns, fp_syn) != (size_t)ns ||
                fread(buf_obs, sizeof(float), ns, fp_obs) != (size_t)ns)
                break;

            for (int it = 0; it < ns; it++) {
                num_total += (double)buf_syn[it] * (double)buf_obs[it];
                den_total += (double)buf_syn[it] * (double)buf_syn[it];
            }
        }

        fclose(fp_syn);
        fclose(fp_obs);
        free(buf_syn);
        free(buf_obs);
    }

    if (den_total < 1.0e-30) {
        if (verbose)
            fprintf(stderr, "source_estimation: WARNING zero global synthetic "
                    "energy, cc1=1\n");
        return 1.0f;
    }

    float cc1 = (float)(num_total / den_total);

    if (verbose)
        fprintf(stderr, "source_estimation: global cc1=%.6f  "
                "<cal,obs>=%.4e  <cal,cal>=%.4e\n",
                cc1, num_total, den_total);

    return cc1;
}
