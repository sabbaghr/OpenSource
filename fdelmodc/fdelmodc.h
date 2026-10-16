#ifndef _fdelmodc_h
#define _fdelmodc_h

#include<stdlib.h>
#include<stdio.h>
#include<math.h>

typedef struct _compType { /* Receiver Type */
	int vz;
	int vx;
	int p;
	int txx;
	int tzz;
	int txz;
	int dxvx;
	int dzvz;
	int pp;
	int ss;
	int ud;
	int q;
} compType;

typedef struct _receiverPar { /* Receiver Parameters */
	char *file_rcv;
	compType type;
	int n;
	int nt;
	int delay;
	int skipdt;
	int max_nrec;
	int *z;
	int *x;
	float *zr;
	float *xr;
	int int_p;
	int int_vx;
	int int_vz;
	int scale;
	int sinkdepth;
	int sinkvel;
	float cp;
	float rho;
} recPar;

typedef struct _snapshotPar { /* Snapshot Parameters */
	char *file_snap;
	char *file_beam;
	compType type;
	int nsnap;
	int delay;
	float t0;
	int skipdt;
	int skipdz;
	int skipdx;
	int nz;
	int nx;
	int z1;
	int z2;
	int x1;
	int x2;
	int vxvztime;
	int beam;
	int withbnd;
} snaPar;

typedef struct _modelPar { /* Model Parameters */
	int iorder;
	int ischeme;
	int grid_dir;
	int sh;
	/* Wavefield reconstruction (Aaker et al. 2020): 0=off, 1=record boundary, 2=inject/reconstruct */
	int wfl_recon;
	int wfl_taper; /* [TAPER] opt-in, default 0: cosine ramp (0 at the end point -> 1 after wfl_taper points) on all sources of the horizontal face(s)/line ends */
	float wfl_edgesign; /* [EDGESIGN] opt-in, default 1: factor applied to ALL five injected sources of the first and last point of each horizontal face/line (-1 = reversed polarity) */
	int wfl_open; /* [OPEN] opt-in, default 0: single-sided reconstruction surface = the TOP face only (z=wfl_z1, x=wfl_x1..wfl_x2); wfl_z2 ignored */
	int wfl_fwd; /* [FWD-ORDER] opt-in, default 0, wfl_recon=2: inject the recorded data in FORWARD time order (retarded / radiating), all five fields negated at load; needed for open surfaces */
	int wfl_tshift; /* [TSHIFT] opt-in, default 0, wfl_recon=2 only: take the traction (force-source) data from sample isam-wfl_tshift; 1 = correct injection timing */
	int wfl_corner2; /* [CORNER2] opt-in, default 0, wfl_recon=1 and 2 (must match): corners appear on all faces (horizontal + vertical normal, weight 1/2 each) */
	int wfl_bwd; /* [BWD-KERNEL] opt-in, default 0: with wfl_recon=2 run the update kernel with dt->-dt (native time reversal) */
	int wfl_vectorial; /* [VECTORIAL] opt-in, default 0, wfl_recon=2 only: Ravasi & Curtis (2013) eq. 8
	                     * "vectorial" (common-practice) wavefield extrapolation -- recorded vx,vz routed
	                     * through the FORCE channel (-> vx,vz update) directly, in place of the traction
	                     * that channel normally carries; NOTHING is injected into txx,tzz,txz. This is the
	                     * reduced, non-reciprocal heuristic the paper contrasts with full TENSORIAL
	                     * injection (both traction AND velocity, the default/existing behaviour), and
	                     * which they show introduces bidirectional (up+down) injection and P/S mode
	                     * mixing artifacts not present in the exact formula. */
	float wfl_x1, wfl_x2, wfl_z1, wfl_z2; /* Omega rectangle, absolute model coordinates */
	char *wfl_file; /* boundary wavefield file (written in record mode, read in inject mode) */
	char *file_cp;
	char *file_ro;
	char *file_cs;
	char *file_qp;
	char *file_qs;
	float dz;
	float dx;
	float dt;
	float tmod;
	int nt;
	float z0;
	float x0;
	float t0;
	/* medium max/min values */
	float cp_min;
	float cp_max;
	float cs_min;
	float cs_max;
	float ro_min;
	float ro_max;
	int nz;
	int nx;
	int naz;
	int nax;
	/* Vx: rox */
	int ioXx;
	int ioXz;
	int ieXx;
	int ieXz;
	/* Vz: roz */
	int ioZx;
	int ioZz;
	int ieZx;
	int ieZz;
	/* P, Txx, Tzz: lam, l2m */
	int ioPx;
	int ioPz;
	int iePx;
	int iePz;
	/* Txz: muu */
	int ioTx;
	int ioTz;
	int ieTx;
	int ieTz;
	/* attenuation / dissipative medium */
	float Qp;
	float Qs;
	float *fw;
	float qr;
	int   nfw;
	float *weight;
} modPar;

typedef struct _waveletPar { /* Wavelet Parameters */
	char *file_src; /* general source */
	int nsrcf;
	int nt;
	int ns;
	int nx;
	float dt;
	float ds;
	float fmax;
	int random;
	int seed;
	int nst;
	size_t *nsamp;
	float *cp;
} wavPar;

typedef struct _sourcePar { /* Source Array Parameters */
	int n;
	int type;
	int orient;
	int *z;
	int *x;
	float Mxx;
	float Mzz;
	float Mxz;
	int single;	
	int plane;
	int circle;
	int array;
	int random;
	float *tbeg;
	float *tend;
	int multiwav;
	float angle;
	float velo;
	float amplitude;
	float dip;
	float strike;
	int distribution;
	int window;
    int injectionrate;
	int sinkdepth;
	int src_at_rcv; /* Indicates that wavefield should be injected at receivers */
    float wx;
    float wz;
} srcPar;

typedef struct _shotPar { /* Shot Parameters */
	int n;
	int *z;
	int *x;
} shotPar;

typedef struct _boundPar { /* Boundary Parameters */
	int top;
	int bot;
	int lef;
	int rig;
	float *tapz;
	float *tapx;
	float *tapxz;
	int cfree;
	int *surface;
    int npml;
    float R; /* reflection at side of model */
    float m; /* scaling order */
    float speed; /* speed of moving free surface */
    int spskip;
    int topadd;
    float *pml_Vx;
    float *pml_nzVx;
    float *pml_nxVz;
    float *pml_nzVz;
    float *pml_nxP;
    float *pml_nzP;

} bndPar;


#if __STDC_VERSION__ >= 199901L
  /* "restrict" is a keyword */
#else
#define restrict
#endif

#endif /* _fdelmodc_h */

