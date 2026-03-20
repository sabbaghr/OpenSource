#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "fdelfwi.h"
#include "segy.h"

void vmess(char *fmt, ...);
void verr(char *fmt, ...);

/* Write a 1D float array as a single-trace SU file */
static void write_su_trace(const char *fname, const float *data, int ns,
                           float dt_sec)
{
    FILE *fp = fopen(fname, "w");
    segy hdr;
    memset(&hdr, 0, sizeof(segy));
    hdr.ns = ns;
    hdr.dt = (unsigned short)(dt_sec * 1.0e6f + 0.5f);
    hdr.tracl = 1;
    hdr.tracr = 1;
    hdr.trid = 1;
    fwrite(&hdr, 1, 240, fp);
    fwrite(data, sizeof(float), ns, fp);
    fclose(fp);
}

/* Read a single-trace SU file into a float array */
static void read_su_trace(const char *fname, float *data, int ns)
{
    FILE *fp = fopen(fname, "r");
    segy hdr;
    fread(&hdr, 1, 240, fp);
    fread(data, sizeof(float), ns, fp);
    fclose(fp);
}

int main(int argc, char **argv)
{
    /* Parameters */
    float dt = 0.001f;   /* 1 ms */
    int nt = 1024;
    float fp_wav = 10.0f; /* Ricker peak frequency */
    float flo = 2.0f;     /* Bandpass low corner */
    float fhi = 8.0f;     /* Bandpass high corner */

    /* Override from command line */
    if (argc > 1) flo = atof(argv[1]);
    if (argc > 2) fhi = atof(argv[2]);
    if (argc > 3) fp_wav = atof(argv[3]);

    printf("Test parameters: dt=%.4f nt=%d fp=%.1f flo=%.1f fhi=%.1f\n",
           dt, nt, fp_wav, flo, fhi);

    /* Create Ricker wavelet */
    float *wav = (float *)calloc(nt, sizeof(float));
    float *wav_fft = (float *)calloc(nt, sizeof(float));
    float *wav_su = (float *)calloc(nt, sizeof(float));

    float t0 = 0.10f;  /* time delay */
    for (int it = 0; it < nt; it++) {
        float t = (float)it * dt - t0;
        float u = M_PI * fp_wav * t;
        float u2 = u * u;
        wav[it] = (1.0f - 2.0f * u2) * expf(-u2);
    }

    /* Method 1: Our FFT-based wavelet filter */
    memcpy(wav_fft, wav, nt * sizeof(float));
    bandpass_filter_wavelet(wav_fft, nt, dt, flo, fhi);
    write_su_trace("test_wav_fft.su", wav_fft, nt, dt);

    /* Method 2: sufilter on the original wavelet */
    write_su_trace("test_wav_orig.su", wav, nt, dt);

    /* Build the same 4-corner spec that bandpass_filter_sufile would use */
    float bw = fhi - flo;
    float tw = 0.10f * bw;
    if (tw < 0.5f) tw = 0.5f;
    if (tw > 0.5f * bw) tw = 0.5f * bw;
    if (tw < 0.1f) tw = 0.1f;
    float f1 = flo - tw; if (f1 < 0.0f) f1 = 0.0f;
    float f2 = flo;
    float f3 = fhi;
    float f4 = fhi + tw;

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "sufilter < test_wav_orig.su f=%.4f,%.4f,%.4f,%.4f > test_wav_su.su",
             f1, f2, f3, f4);
    printf("sufilter command: %s\n", cmd);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "sufilter failed!\n");
        return 1;
    }

    /* Read back sufilter result */
    read_su_trace("test_wav_su.su", wav_su, nt);

    /* Compare: compute max difference and relative error */
    float max_diff = 0.0f, max_fft = 0.0f, max_su = 0.0f;
    double rms_diff = 0.0, rms_fft = 0.0;
    for (int it = 0; it < nt; it++) {
        float d = fabsf(wav_fft[it] - wav_su[it]);
        if (d > max_diff) max_diff = d;
        if (fabsf(wav_fft[it]) > max_fft) max_fft = fabsf(wav_fft[it]);
        if (fabsf(wav_su[it]) > max_su) max_su = fabsf(wav_su[it]);
        rms_diff += (double)(wav_fft[it] - wav_su[it]) * (wav_fft[it] - wav_su[it]);
        rms_fft += (double)wav_fft[it] * wav_fft[it];
    }
    rms_diff = sqrt(rms_diff / nt);
    rms_fft = sqrt(rms_fft / nt);

    float rel_err = (rms_fft > 1.0e-30) ? (float)(rms_diff / rms_fft) : 0.0f;

    printf("\n=== RESULTS ===\n");
    printf("Corner frequencies: f1=%.2f f2=%.2f f3=%.2f f4=%.2f (taper=%.2f Hz)\n",
           f1, f2, f3, f4, tw);
    printf("Max |FFT filtered|:    %.6e\n", max_fft);
    printf("Max |sufilter|:        %.6e\n", max_su);
    printf("Max |FFT - sufilter|:  %.6e\n", max_diff);
    printf("RMS difference:        %.6e\n", (float)rms_diff);
    printf("Relative error:        %.6e\n", rel_err);

    if (rel_err < 1.0e-3) {
        printf("\nPASS: Filters match (relative error < 0.1%%)\n");
    } else if (rel_err < 1.0e-2) {
        printf("\nWARN: Small filter mismatch (relative error < 1%%)\n");
    } else {
        printf("\nFAIL: Significant filter mismatch (relative error = %.2f%%)\n",
               rel_err * 100.0f);
    }

    /* Write difference trace for visual inspection */
    float *wav_diff = (float *)calloc(nt, sizeof(float));
    for (int it = 0; it < nt; it++)
        wav_diff[it] = wav_fft[it] - wav_su[it];
    write_su_trace("test_wav_diff.su", wav_diff, nt, dt);

    printf("\nOutput files:\n");
    printf("  test_wav_orig.su  - original Ricker wavelet\n");
    printf("  test_wav_fft.su   - FFT-filtered wavelet\n");
    printf("  test_wav_su.su    - sufilter-filtered wavelet\n");
    printf("  test_wav_diff.su  - difference (FFT - sufilter)\n");

    free(wav); free(wav_fft); free(wav_su); free(wav_diff);

    /* Clean up */
    remove("test_wav_orig.su");

    return (rel_err > 1.0e-2) ? 1 : 0;
}
