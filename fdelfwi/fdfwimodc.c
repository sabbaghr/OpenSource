#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"


#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

double wallclock_time(void);

void threadAffinity(void);

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav, srcPar *src, shotPar *shot, bndPar *bnd, int verbose);

int readModel(modPar mod, bndPar bnd, float *rox, float *roz, float *l2m, float *lam, float *muu, float *tss, float *tes, float *tep);

int defineSource(wavPar wav, srcPar src, modPar mod, recPar rec, shotPar shot, float **src_nwav, int reverse, int verbose);

int writeSrcRecPos(modPar *mod, recPar *rec, srcPar *src, shotPar *shot);

int acoustic16(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose);

int acoustic6(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose);

int acoustic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose);

int acoustic4pml(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose);

int acousticSH4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *tx, float *tz, float *vz, float *rox, float *roz, float *mul, int verbose);

int acoustic4_qr(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose);

int acoustic2(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose);

int acoustic4Block(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx,
float *vz, float *p, float *rox, float *roz, float *l2m, int verbose);

int viscoacoustic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, float *tss, float *tep, float *q, int verbose);

int elastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int verbose);

int elastic4dc(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int verbose);

int viscoelastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float
*vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, float *ts, float *tep, float
*tes, float *r, float *q, float *p, int verbose);

int elastic6(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc,
    float **src_nwav, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox,
    float *roz, float *l2m, float *lam, float *mul, int verbose);

int getRecTimes(modPar mod, recPar rec, bndPar bnd, int itime, int isam, float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *q, float *l2m, float *lam, float *rox, float *roz,
    float *rec_vx, float *rec_vz, float *rec_txx, float *rec_tzz, float *rec_txz,
    float *rec_p, float *rec_pp, float *rec_ss, float *rec_q, float *rec_udp, float *rec_udvz, float *rec_dxvx, float *rec_dzvz, int verbose);

int writeRec(recPar rec, modPar mod, bndPar bnd, wavPar wav, int ixsrc, int izsrc, int nsam, int ishot, int nshots, int fileno,
			 float *rec_vx, float *rec_vz, float *rec_txx, float *rec_tzz, float *rec_txz,
			 float *rec_p, float *rec_pp, float *rec_ss, float *rec_q, float *rec_udp, float *rec_udvz, float *rec_dxvx, float *rec_dzvz, int verbose);

int writeSnapTimes(modPar mod, snaPar sna, bndPar bnd, wavPar wav,int ixsrc, int izsrc, int itime,
				   float *vx, float *vz, float *tzz, float *txx, float *txz, int verbose);

int getBeamTimes(modPar mod, snaPar sna, float *vx, float *vz, float *tzz, float *txx, float *txz,
				 float *beam_vx, float *beam_vz, float *beam_txx, float *beam_tzz, float *beam_txz,
				 float *beam_p, float *beam_pp, float *beam_ss, int verbose);

int writeBeams(modPar mod, snaPar sna, int ixsrc, int izsrc, int ishot, int fileno,
			   float *beam_vx, float *beam_vz, float *beam_txx, float *beam_tzz, float *beam_txz,
			   float *beam_p, float *beam_pp, float *beam_ss, int verbose);

int writeCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl);

int allocStoreSourceOnSurface(srcPar src);

int freeStoreSourceOnSurface(void);

int fdfwimodc(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd, recPar *rec, snaPar *sna,
              int ixsrc, int izsrc, float **src_nwav, int ishot, int nshots, int fileno,
              checkpointPar *chk, int verbose)
{
	wflPar wfl;
	int it, isam;
	size_t sizem, size;
	size_t perc;
	double t3;

	/* Calculate time loop limits */
	/* Add extra steps for negative start time (e.g. Ricker wavelet) */
	int it0 = 0;
	int it1 = mod->nt + NINT(-mod->t0 / mod->dt);

	/* ------------------------------------------------ */
	/* 1. Allocate Wavefields                           */
	/* ------------------------------------------------ */
	sizem = mod->nax * mod->naz;

	/* Initialize struct to zero to avoid garbage pointers */
	memset(&wfl, 0, sizeof(wflPar));

	wfl.vx  = (float *)calloc(sizem, sizeof(float));
	wfl.vz  = (float *)calloc(sizem, sizeof(float));

	/* For acoustic, tzz stores Pressure (P) */
	wfl.tzz = (float *)calloc(sizem, sizeof(float));

	if (mod->ischeme > 2) {
		wfl.txz = (float *)calloc(sizem, sizeof(float));
		wfl.txx = (float *)calloc(sizem, sizeof(float));
	}

	if (mod->ischeme == 2 || mod->ischeme == 4) {
		wfl.r = (float *)calloc(sizem, sizeof(float));
		wfl.q = (float *)calloc(sizem, sizeof(float));
		if (mod->ischeme == 4) wfl.p = (float *)calloc(sizem, sizeof(float));
	}

	isam = 0;

	/* Preserve the original fileno (shot index) for FWI-style output.
	 * The time loop uses fileno internally for time-segment bookkeeping
	 * (itwritten = fileno * rec->nt * rec->skipdt), so it must start at 0.
	 * fileno_shot stores the original shot index for output file naming. */
	int fileno_shot = fileno;
	fileno = 0;

	/* Reset checkpoint counter */
	if (chk) chk->it = 0;

	/* ------------------------------------------------ */
	/* 2. Allocate Receiver Buffers                     */
	/* ------------------------------------------------ */
	float *rec_vx=NULL, *rec_vz=NULL, *rec_p=NULL;
	float *rec_txx=NULL, *rec_tzz=NULL, *rec_txz=NULL;
	float *rec_pp=NULL, *rec_ss=NULL, *rec_q=NULL;
	float *rec_udp=NULL, *rec_udvz=NULL;
	float *rec_dxvx=NULL, *rec_dzvz=NULL;

	size = rec->n * rec->nt;
	if (size == 0) {
		free(wfl.vx);
		free(wfl.vz);
		free(wfl.tzz);
		if (wfl.txz) free(wfl.txz);
		if (wfl.txx) free(wfl.txx);
		if (wfl.r)   free(wfl.r);
		if (wfl.q)   free(wfl.q);
		if (wfl.p)   free(wfl.p);
		return 0;
	}

	if (rec->type.vx)   rec_vx   = (float *)calloc(size, sizeof(float));
	if (rec->type.vz)   rec_vz   = (float *)calloc(size, sizeof(float));
	if (rec->type.p)    rec_p    = (float *)calloc(size, sizeof(float));
	if (rec->type.txx)  rec_txx  = (float *)calloc(size, sizeof(float));
	if (rec->type.tzz)  rec_tzz  = (float *)calloc(size, sizeof(float));
	if (rec->type.txz)  rec_txz  = (float *)calloc(size, sizeof(float));
	if (rec->type.dxvx) rec_dxvx = (float *)calloc(size, sizeof(float));
	if (rec->type.dzvz) rec_dzvz = (float *)calloc(size, sizeof(float));
	if (rec->type.pp)   rec_pp   = (float *)calloc(size, sizeof(float));
	if (rec->type.ss)   rec_ss   = (float *)calloc(size, sizeof(float));
	if (rec->type.q)    rec_q    = (float *)calloc(size, sizeof(float));
	if (rec->type.ud) {
		rec_udvz = (float *)calloc(mod->nax * rec->nt, sizeof(float));
		rec_udp  = (float *)calloc(mod->nax * rec->nt, sizeof(float));
	}

	/* ------------------------------------------------ */
	/* 3. Time-stepping loop                            */
	/* ------------------------------------------------ */
	perc = it1 / 100;
	if (!perc) perc = 1;

	if (verbose) {
		fprintf(stderr, "    %s: Progress: %3d%%", xargv[0], 0);
	}

	for (it=it0; it<it1; it++) {

#pragma omp parallel default(shared)
{
		if (it==it0 && verbose>2) {
			threadAffinity();
		}
		switch ( mod->ischeme ) {
			case -1 : /* Acoustic dissipative media FD kernel */
				acoustic4_qr(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl.vx, wfl.vz, wfl.tzz, mod->rox, mod->roz, mod->l2m, verbose);
				break;
			case 1 : /* Acoustic FD kernel */
				if (mod->iorder==2) {
					acoustic2(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
						wfl.vx, wfl.vz, wfl.tzz, mod->rox, mod->roz, mod->l2m, verbose);
				}
				else if (mod->iorder==4) {
					if (mod->sh) {
						acousticSH4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
							wfl.vx, wfl.vz, wfl.tzz, mod->rox, mod->roz, mod->l2m, verbose);
					}
					else {
						acoustic4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
							wfl.vx, wfl.vz, wfl.tzz, mod->rox, mod->roz, mod->l2m, verbose);
					}
				}
				else if (mod->iorder==6) {
					acoustic6(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
						wfl.vx, wfl.vz, wfl.tzz, mod->rox, mod->roz, mod->l2m, verbose);
				}
				else if (mod->iorder==16) {
					acoustic16(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
						wfl.vx, wfl.vz, wfl.tzz, mod->rox, mod->roz, mod->l2m, verbose);
				}
				break;
			case 2 : /* Visco-Acoustic FD kernel */
				viscoacoustic4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl.vx, wfl.vz, wfl.tzz, mod->rox, mod->roz, mod->l2m,
					mod->tss, mod->tep, wfl.q, verbose);
				break;
			case 3 : /* Elastic FD kernel */
				if (mod->iorder==4) {
					elastic4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
						wfl.vx, wfl.vz, wfl.tzz, wfl.txx, wfl.txz,
						mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
				}
				else if (mod->iorder==6) {
					elastic6(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
						wfl.vx, wfl.vz, wfl.tzz, wfl.txx, wfl.txz,
						mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
				}
				break;
			case 4 : /* Visco-Elastic FD kernel */
				viscoelastic4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl.vx, wfl.vz, wfl.tzz, wfl.txx, wfl.txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu,
					mod->tss, mod->tep, mod->tes, wfl.r, wfl.q, wfl.p, verbose);
				break;
			case 5 : /* Elastic FD kernel with S-velocity set to zero*/
				elastic4dc(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl.vx, wfl.vz, wfl.tzz, wfl.txx, wfl.txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
				break;
		}

		/* write samples to file if rec->nt samples are calculated */

#pragma omp master
{
		if ( (((it-rec->delay) % rec->skipdt)==0) && (it >= rec->delay) ) {
			int writeToFile, itwritten;

			if ((((it-rec->delay+NINT(mod->t0/mod->dt))/rec->skipdt)+1)!=0) {
				writeToFile = ! ( (((it-rec->delay+NINT(mod->t0/mod->dt))/rec->skipdt)+1)%rec->nt );
			}
			else { /* when negative times passes zero-time */
				writeToFile = 0;
			}
			itwritten   = fileno*(rec->nt)*rec->skipdt;
			/* Note that time step it=0 (t=0 for t**-fields t=-1/2 dt for v*-field) is not recorded */
			/* negative time correction with mod->t0 for dipping plane waves modeling */
			isam        = (it-rec->delay-itwritten+NINT(mod->t0/mod->dt))/rec->skipdt+1;
			if (isam < 0) isam = rec->nt+isam;
			/* store time at receiver positions */
			getRecTimes(*mod, *rec, *bnd, it, isam,
				wfl.vx, wfl.vz, wfl.tzz, wfl.txx, wfl.txz, wfl.q,
				mod->l2m, mod->lam, mod->rox, mod->roz,
				rec_vx, rec_vz, rec_txx, rec_tzz, rec_txz,
				rec_p, rec_pp, rec_ss, rec_q, rec_udp, rec_udvz, rec_dxvx, rec_dzvz, verbose);

			/* at the end of modeling a shot, write receiver array to output file(s) */
			if (writeToFile && (it+rec->skipdt <= it1-1) ) {
				fileno = ( ((it-rec->delay)/rec->skipdt)+1)/rec->nt;
				writeRec(*rec, *mod, *bnd, *wav, ixsrc, izsrc, isam+1, ishot, nshots, fileno,
					rec_vx, rec_vz, rec_txx, rec_tzz, rec_txz,
					rec_p, rec_pp, rec_ss, rec_q, rec_udp, rec_udvz, rec_dxvx, rec_dzvz, verbose);
			}
		}

		/* write snapshots to output file(s) */
		if (sna->nsnap) {
			writeSnapTimes(*mod, *sna, *bnd, *wav, ixsrc, izsrc, it,
				wfl.vx, wfl.vz, wfl.tzz, wfl.txx, wfl.txz, verbose);
		}

		/* write FWI checkpoint (complete wavefield state) */
		if (chk && chk->nsnap > 0) {
			if ( (((it - chk->delay) % chk->skipdt) == 0) &&
			     (it >= chk->delay) &&
			     (chk->it < chk->nsnap) ) {
				if (verbose > 1)
					vmess("fdfwimodc: Writing checkpoint %d/%d at it=%d", chk->it+1, chk->nsnap, it);
				writeCheckpoint(chk, chk->it, &wfl);
				chk->it++;
			}
		}
}
#pragma omp barrier

#pragma omp master
{
		if (verbose) {
			if(!((it1-it)%perc)) fprintf(stderr,"\b\b\b\b%3d%%",it*100/(it1-it0));
			if(it==100) t3=wallclock_time();
			if(it==500){
				t3=(wallclock_time()-t3)*((it1-it0)/400.0);
				fprintf(stderr,"\r    %s: Estimated total compute time for this shot = %.2fs.\n    %s: Progress: %3d%%",
					xargv[0],t3,xargv[0],it/((it1-it0)/100));
			}
		}
}
} /* end of OpenMP parallel section */

	} /* end of loop over time steps it */

	/* write output files: receivers */
	/* For FWI: use fileno_shot to create separate files per shot */
	/* For traditional use: fileno tracks time-segment file splitting */
	if (fileno) fileno++;

	if (rec->scale==1) { /* scale receiver with distance src-rcv */
		float xsrc, zsrc, Rrec, rdx, rdz;
		int irec;
		xsrc=mod->x0+mod->dx*ixsrc;
		zsrc=mod->z0+mod->dz*izsrc;
		for (irec=0; irec<rec->n; irec++) {
			rdx=mod->x0+rec->xr[irec]-xsrc;
			rdz=mod->z0+rec->zr[irec]-zsrc;
			Rrec = sqrt(rdx*rdx+rdz*rdz);
			fprintf(stderr,"Rec %d is scaled with distance %f R=%.2f,%.2f S=%.2f,%.2f\n", irec, Rrec,rdx,rdz,xsrc,zsrc);
			for (it=0; it<rec->nt; it++) {
				rec_p[irec*rec->nt+it] *= sqrt(Rrec);
			}
		}
	}
	/* Use fileno_shot (preserved shot index) for FWI-style per-shot files */
	/* Reset ishot to 0 for each shot to avoid append mode in writeRec */
	writeRec(*rec, *mod, *bnd, *wav, ixsrc, izsrc, isam+1, ishot, nshots, fileno_shot,
		rec_vx, rec_vz, rec_txx, rec_tzz, rec_txz,
		rec_p, rec_pp, rec_ss, rec_q, rec_udp, rec_udvz, rec_dxvx, rec_dzvz, verbose);

	if (verbose) {
		fprintf(stderr,"\b\b\b\b%3d%%\n",100);
	}

	/* ------------------------------------------------ */
	/* 4. Free wavefield arrays                         */
	/* ------------------------------------------------ */
	free(wfl.vx);
	free(wfl.vz);
	free(wfl.tzz);
	if (wfl.txz) free(wfl.txz);
	if (wfl.txx) free(wfl.txx);
	if (wfl.r)   free(wfl.r);
	if (wfl.q)   free(wfl.q);
	if (wfl.p)   free(wfl.p);

	/* Free receiver arrays */
	if (rec_vx)   free(rec_vx);
	if (rec_vz)   free(rec_vz);
	if (rec_p)    free(rec_p);
	if (rec_txx)  free(rec_txx);
	if (rec_tzz)  free(rec_tzz);
	if (rec_txz)  free(rec_txz);
	if (rec_dxvx) free(rec_dxvx);
	if (rec_dzvz) free(rec_dzvz);
	if (rec_pp)   free(rec_pp);
	if (rec_ss)   free(rec_ss);
	if (rec_q)    free(rec_q);
	if (rec_udvz) free(rec_udvz);
	if (rec_udp)  free(rec_udp);

	return 0;
}
