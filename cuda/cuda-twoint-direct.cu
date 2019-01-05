#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "ofmo-def.h"
#ifdef __cplusplus
}
#endif
#include "cudalib.h"
#include "cuda-twoint-direct.h"
#include "cuda-twoint-core.h"
#include "cuda-twoint-core-os.h"
#include "cuda-twoint-core-rys.h"
#include "cuda-utils.h"
#ifdef CUDA_FMT_M
#include "cuda-fmt-m.h"
#endif

/*
//#define EPS_PS4 1.e-30
//#define EPS_ERI	1.e-15
//#define EPS_PS4 1.e-20
#define EPS_PS4 1.e-20F
#define EPS_ERI	1.e-15
//#define EPS_SCH	5.e-11
#define EPS_SCH	1e-12F
*/

#ifndef ZERO
#define ZERO	0.e0
#endif
#ifndef ONE
#define ONE	1.e0
#endif
#ifndef HALF
#define HALF	.5e0
#endif

#ifndef MIN2
#define MIN2(a, b) (((a)<(b))? (a): (b))
#endif
#ifndef MAX2
#define MAX2(a, b) (((a)>(b))? (a): (b))
#endif

#define NKLBUF 8

extern __shared__ double shared[];
//__device__ int ijcounter[21];

#ifndef WORK
#define WORK(a) work[nthb*(a)+tidx]
#endif

//#if CUDA_ARCH >= 350
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
#define NREGS_064 __launch_bounds__(128,8)
#define NREGS_128 __launch_bounds__(128,4)
#define NREGS_255 __launch_bounds__(128,2)
#else
#define NREGS_064
#define NREGS_128
#define NREGS_255
#endif

//#define GPU_TWOINT_OS
//#define GPU_TWOINT_RYS

/* ------------------------------------- */

__global__ void gpu_twoint_direct_counter_init(int Labcd, int counter)
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int bidx = blockIdx.x;
  __syncthreads();
  __threadfence();
  if (bidx==0 && tidx==0) {
    ijcounter[Labcd] = counter;
  }
  __syncthreads();
  __threadfence();

  return;
}

/* ------------------------------------- */

__device__ float gpu_dmax6(const int i, const int j,
    const int k, const int l)
{
  float dmax = 1.0;
  /*
  int i2 = i * ncs;
  int j2 = j * ncs;
  int k2 = k * ncs;
  dmax = MAX2(Dcs[i2+j], Dcs[k2+l]) * 4;
  dmax = MAX2(Dcs[i2+k], dmax);
  dmax = MAX2(Dcs[i2+l], dmax);
  dmax = MAX2(Dcs[j2+k], dmax);
  dmax = MAX2(Dcs[j2+l], dmax);
  */
  int i2 = i * ncs;
  dmax = MAX2(LDG(Dcs[i2+j]), LDG(Dcs[k*ncs+l])) * 4;
  dmax = MAX2(LDG(Dcs[i2+k]), dmax);
  dmax = MAX2(LDG(Dcs[i2+l]), dmax);
  i2 = j * ncs;
  dmax = MAX2(LDG(Dcs[i2+k]), dmax);
  dmax = MAX2(LDG(Dcs[i2+l]), dmax);

  return dmax;
}

/* ------------------------------------- */

/** (ss,ss)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
NREGS_064
gpu_twoint_direct_ssss_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=0, Lb=0, Lc=0, Ld=0;
    int Lab, Lcd;
    int Labcd = 0;
    int *p_ijcounter = &ijcounter[Labcd];
    int ijcs, ijcs0, ijcs1, ijao;
//    int klcs, klcs0, klcs1, klao;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
//    int ics, iat, iao, jcs, jat, jao, kcs, kat, kao, lcs, lat, lao;
    int ics, iat, iao, jcs, jat, jao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
    __shared__ double BA[3];
//    __shared__ double A[3];
//    __shared__ double sSSSS[NTHREADS];
//    __shared__ double sGij[NTHREADS];
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[0];
    double *tbl = FMT_m_table0;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    //double SSSS[1];
    //double Gij[1];
    //double *SSSS = &shared[tsize+1+tidx];
    //double *Gij = &shared[tsize+1+tidx+nthb];
    size_t sOffset = tsize;
    double *Gij = &shared[tsize+tidx];
    double *SSSS = &shared[tsize+tidx+nthb];
    tsize += nthb*2;
    //double *Gij = &shared[tsize+1+tidx];
#else
    size_t sOffset = 0;
    double *Gij = &shared[tidx];
    double *SSSS = &shared[nthb+tidx];
    size_t tsize = nthb*3;
#endif
#ifdef DLB_KL_SSSS
    //__shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
    //volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif

    mincs = MIN2(leading_cs[Lc], leading_cs[Ld]);
    minao = shel_ini[mincs];
    maxcs = MAX2(leading_cs[Lc+1], leading_cs[Ld+1]) - 1;
    maxao = shel_ini[maxcs+1]-1;
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
    Gkl = Gkl_b + bidx * num_Gkl;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao;
    work = work_b + bidx * nthb * WORK_SIZE;

//    nwkblk = nworkers * ndev * nblk;
//    iwkblk = workerid * ndev * nblk + idev * nblk + bidx;
//    nwkblk = nwks;
//    iwkblk = iwk + bidx;
    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;
//    if(tidx==0) printf("ssss:(%3d/%3d) %d %d %d %16.10g\n", iwkblk, nwkblk, workerid, idev, bidx, G[0]);

    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lab];
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
//    if(tidx==0) printf("ssss:(%1d/%1d) ijcs0 = %d\n", iwkblk, nwkblk, ijcs0);

#ifndef USE_ATOMIC
    /*
    for ( i=0; i<nao; i++ ) Gi[i]=ZERO;
    for ( i=0; i<nao; i++ ) Gj[i]=ZERO;
    */
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*2; i+=nthb) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0); i+=nthb ) Gkl[i]=ZERO;
    __syncthreads();
#endif

//    if (tidx==0) ijcsw = ijcs0 + atomicAdd(p_ijcounter,1);
#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx;
      nijcsw = 0;
    }
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (ijcs=ijcs0+iwkblk; ijcs<ijcs1; ijcs+=nwkblk ) {
#endif
        int klcs0a;
//        int n2ei = 0;
//        double Gij;
//        double A[3],BA[3];
        double A[3];
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs1 - 1 - (ijcsw - ijcs0);
        /*
        __syncthreads();
        if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
        */
#endif
#ifdef DLB_KL_SSSS
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
        klcs0a = klcs0;
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else
        {
	  float val_ab = csp_schwarz[ijcs];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
//	  for (int klcs = klcs0a+tidx ; klcs<=ijcs; klcs+=nthb) {
	  for (int klcs = ijcs-tidx; klcs>=klcs0a ; klcs-=nthb) {
//	  for (int klcs = klcs0a ; klcs<=ijcs; klcs++) {
              int kcs, lcs;
              int iklcs;
              float dmax;
	      float val_abcd = val_ab * csp_schwarz[klcs];
	      if ( val_abcd < eps_ps4 ) continue;
	      kcs    = csp_ics[klcs];
	      lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	      if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
//          if(bidx==0&&tidx==0) printf("nklcs = %d\n", nklcs);
        }
#endif
//        Gij = ZERO;
//        sGij[tidx] = ZERO;
        *Gij = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao    = shel_ini[ics];
	jao    = shel_ini[jcs];
	ijao   = iao*(iao+1)/2+jao;
        /*
        {
          double B[3];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
#pragma unroll
//	for ( i=tidx; i<3; i+=nthb ) BA[i] = B[i] - A[i];
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
  __threadfence_block();
        }
        */
        __syncthreads();
        BA[0] = LDG(atom_x[jat]) - (A[0] = LDG(atom_x[iat]));
        BA[1] = LDG(atom_y[jat]) - (A[1] = LDG(atom_y[iat]));
        BA[2] = LDG(atom_z[jat]) - (A[2] = LDG(atom_z[iat]));
	
//	for ( ; klcs<=ijcs; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs = klcs0a+tidx ; klcs<=ijcs; klcs+=nthb ) {
#else
#ifndef DLB_KL_SSSS
        for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
#else
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
          while(iklcs<nklcs) {
#endif /* DLB_KL */
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao, lcs, lat, lao;
            int klps0, nklps;
            int klao;
            int klcs2;
#ifdef USE_INSTANT_SCHWARZ
            {
	    float val_abcd = val_ab * csp_schwarz[klcs];
            float dmax;
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
            }
#else
            int klcs;
            klcs = sklcs[iklcs];
#endif
            klcs2 = klcs - klcs0;
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao    = LDG(shel_ini[kcs]);
	    lao    = LDG(shel_ini[lcs]);
	    klao   = kao*(kao+1)/2+lao;
            {
            /*
            double C[3], D[3];
            double DC[3],AC[3];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
            */
            double dtmp;
            double DC[3],AC[3];
            AC[0] = A[0] - (dtmp = LDG(atom_x[kat]));
            DC[0] = LDG(atom_x[lat]) - dtmp;
            AC[1] = A[1] - (dtmp = LDG(atom_y[kat]));
            DC[1] = LDG(atom_y[lat]) - dtmp;
            AC[2] = A[2] - (dtmp = LDG(atom_z[kat]));
            DC[2] = LDG(atom_z[lat]) - dtmp;

	    gpu_twoint_core_ssss_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      SSSS );
//		    &psp_xiza[klps0], DC,   AC,      &sSSSS[tidx] );
            }
//	    if ( fabs(sSSSS[tidx]) > eps_eri ) {
	    if ( fabs(SSSS[0]) > eps_eri ) {
		double x, x4;
		int ij, ik, il, jk, jl, kl, i0, j0;
                double coe;
                double dx;
		coe = ONE;
		if ( iao == jao ) coe = HALF;
		if ( kao == lao ) coe *= HALF;
		if ( ijao == klao ) coe *= HALF;
		x  = coe * SSSS[0];
		x4 = 4.0 * x;
                x *= -x_coef;
//		x  = coe * sSSSS[tidx];

		i0 = iao*nao;
		j0 = jao*nao;
		ij = i0 + jao;
		ik = i0 + kao;
		il = i0 + lao;
		jk = j0 + kao;
		jl = j0 + lao;
		kl = kao*nao + lao;
                /*
		G[ij] += x4*Ds[kl];
		G[kl] += x4*Ds[ij];
		G[ik] -=  x*Ds[jl];
		G[il] -=  x*Ds[jk];
		G[jk] -=  x*Ds[il];
		G[jl] -=  x*Ds[ik];
                */
#ifdef USE_ATOMIC
                dx = x*LDG(Ds[jl]);
                atomicAdd(&G[ik],dx);
                dx = x*LDG(Ds[jk]);
                atomicAdd(&G[il],dx);
                dx = x*LDG(Ds[il]);
                atomicAdd(&G[jk],dx);
                dx = x*LDG(Ds[ik]);
                atomicAdd(&G[jl],dx);

                *Gij += x4*LDG(Ds[kl]);
                dx = x4*LDG(Ds[ij]);
                atomicAdd(&G[kl],dx);
#else
                Gi[kao] += x*LDG(Ds[jl]);
                Gi[lao] += x*LDG(Ds[jk]);
                Gj[kao] += x*LDG(Ds[il]);
                Gj[lao] += x*LDG(Ds[ik]);

                Gkl[klcs2] += x4*LDG(Ds[ij]);
                *Gij += x4*LDG(Ds[kl]);
#endif
//                n2ei++;
	    }	// if ( fabs(SSSS[0]) > eps_eri )
#ifdef DLB_KL_SSSS
          if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
          iklcs = klcsw[widx] + tidw;
          }; // while (iklcs)
#endif
	}	// for ( klcs );
//    if(tidx==0&&iwkblk==1) printf("ssss:(%1d/%1d) %d %d %d\n", iwkblk, nwkblk, ijcs, nklcs, n2ei);
//	klcs = klcs0;
#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
#if 0
        atomicAdd(&G[iao*nao+jao], *Gij);
#else
          { // Reduce Gij within warp
            //double *sGij = &shared[widx*WARP_SIZE];
            double *sGij = &shared[sOffset+widx*WARP_SIZE];
            warpReduce(sGij, tidw);
          }
          __syncthreads();
          if (tidx==0) { // Reduce Gij for warp masters
            //double *sGij = &shared[0];
            double *sGij = &shared[sOffset];
            double dtmp = sGij[0];
            for (int j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
            atomicAdd(&G[iao*nao+jao], dtmp);
          }
#endif
#else 
//        Gi[jao] += sGij[tidx];
        Gi[jao] += *Gij;
        {
          int nGi = num_Gi*2;
          int bs = bidx*nthb*num_Gi;
          int i0 = iao*nao;
          int j0 = jao*nao;
          int i,j;
          int ijcs_next;
          int ics_next=-1, jcs_next=-1;
          __syncthreads();
#ifdef GPU_DLB
//          ijcs_next = ijcsw;
//          ijcs_next = ijcs1 - ijcsw + 1;
          ijcs_next = ijcs1 - 1 - (ijcsw - ijcs0);
          if (ijcsw<ijcs1) { 
#else
          ijcs_next = ijcs+nwkblk;
          if (ijcs_next<ijcs1) { 
#endif
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
          if (ics != ics_next) {
          for (i=minao+tidx;i<=maxao;i+=nthb) {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
            /*
            for (j=0;j<nthb;j++) {
              Gtmp += *pG;
              *pG = ZERO;
              pG  += num_Gi;
            }
            */
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
          } 
          }
          if (jcs != jcs_next) {
          for (i=minao+tidx;i<=maxao;i+=nthb) {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao + i;
            double *pG1 = pG + num_Gi;
            /*
            for (j=0;j<nthb;j++) {
              Gtmp += *pG;
              *pG = ZERO;
              pG  += num_Gi;
            }
            */
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          }
          } 
        }
#endif /* USE_ATOMIC */
        __syncthreads();
    }		// for ( ijcs );
    __syncthreads();
#ifndef USE_ATOMIC
    for (int klcs = klcs0+tidx ; klcs<klcs1; klcs+=nthb ) {
      int kcs,lcs,kao,lao;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs];
      lao = shel_ini[lcs];
//      G[kao*nao+lao] += Gkl[klcs];
      G[kao*nao+lao] += Gkl[klcs-klcs0];
    }
#endif

    __syncthreads();
//    if(tidx==0) printf("ssss:(%3d/%3d) %d %d %d %16.10g\n", iwkblk, nwkblk, workerid, idev, bidx, G[0]);

    return;
}

/** (ps,ss)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
NREGS_128
gpu_twoint_direct_psss_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=1, Lb=0, Lc=0, Ld=0;
    int Lab, Lcd;
    int Labcd = 1;
    int *p_ijcounter = &ijcounter[Labcd];
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, klcs0a;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, lcs, lat, lao;
    double PSSS[3];
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
//    double coe;
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    int ics2, jcs2;
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[1];
    double *tbl = FMT_m_table1;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    //double Gij[3];
    size_t sOffset = tsize;
    double *Gij = &shared[tsize+tidx*3];
    tsize += nthb*3;
#else
    size_t sOffset = 0;
    double *Gij = &shared[tidx*3];
    size_t tsize = nthb*3;
#endif
#ifdef DLB_KL_PSSS
//    __shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
//    volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif
    __shared__ double A[3];
    __shared__ double BA[3];

    mincs = MIN2(leading_cs[Lc], leading_cs[Ld]);
    minao = shel_ini[mincs];
    maxcs = MAX2(leading_cs[Lc+1], leading_cs[Ld+1]) - 1;
    maxao = shel_ini[maxcs+1]-1;
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;
//    if (bidx == 0 && tidx == 0)
//      printf("mincs(minao), maxcs(maxao) = %4d(%4d), %4d(%4d)\n", mincs, minao, maxcs, maxao); 

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
//    Gkl = Gkl_b + bidx * ncspair * nL[maxlqn] * nL[maxlqn];
    Gkl = Gkl_b + bidx * num_Gkl;
//    Gi = (double *)malloc(nao * sizeof(double) * 4);
//    Gj = Gi + nao*3;
//    Gi   = Gi_t + (bidx * nthb + tidx) * nL[maxlqn] * nao;
//    Gj   = Gj_t + (bidx * nthb + tidx) * nL[maxlqn] * nao;
//    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
//    Gj   = Gj_t + (bidx * nthb + tidx) * num_Gi;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao*3;
    work = work_b + bidx * nthb * WORK_SIZE;

//    nwkblk = nworkers * ndev * nblk;
//    iwkblk = workerid * ndev * nblk + idev * nblk + bidx;
//    nwkblk = nwks;
//    iwkblk = iwk + bidx;
    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;
//    if(tidx==0) printf("psss:(%3d/%3d) %d %d %d\n", iwkblk, nwkblk, workerid, idev, bidx);

    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lab];
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];

#ifndef USE_ATOMIC
    /*
    for ( i=0; i<nao*3; i++ ) Gi[i]=ZERO;
    for ( i=0; i<nao; i++ ) Gj[i]=ZERO;
    */
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*4; i+=nthb ) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0); i+=nthb ) Gkl[i]=ZERO;
    __syncthreads();
#endif

#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
//    ijcsw = ijcs0 + iwk + nwks * bidx;
//    nijcsw = 0;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx * NIJCSW;
      nijcsw = NIJCSW-1;
    }
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (int ijcs2=ijcs0+iwkblk; ijcs2<ijcs1; ijcs2+=nwkblk ) {
#endif
//        double Gij[3];
//        double A[3],BA[3];
//        double A[3];
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs0 + ijcs1 - ijcsw - 1;
        /*
        __syncthreads();
        if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
        */
#else
      ijcs = ijcs2;
#endif
#ifdef SORT_IJ_SCHWARZ
      ijcs = sorted_csp[ijcs];
#endif
#ifdef DLB_KL_PSSS
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
        klcs0a = klcs0;
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else
        {
	  float val_ab = csp_schwarz[ijcs];
//          int sklcs_l[NKLBUF];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
	  for (int klcs2 = klcs0a+tidx ; klcs2<klcs1; klcs2+=nthb) {
//	  for (int klcs2 = klcs1-tidx-1; klcs2>=klcs0a ; klcs2-=nthb) {
              int klcs=klcs2;
              int kcs, lcs;
              int iklcs;
              float dmax;
	      float val_abcd;
#ifdef SORT_KL_SCHWARZ
          klcs = sorted_csp[klcs2];
#endif
	      val_abcd = val_ab * csp_schwarz[klcs];
	      if ( val_abcd < eps_ps4 ) continue;
	      kcs    = csp_ics[klcs];
	      lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	      if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
#ifdef SORT_KL_SCHWARZ
              klcs = klcs2;
#endif
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
//          if(bidx==0&&tidx==0) printf("nklcs = %d\n", nklcs);
        }
#endif /* USE_INSTANT_SCHWARZ */
#pragma unroll
        for (int i=0;i<3;i++) Gij[i] = ZERO;
//        for (i=0;i<3;i++) sGij[i][tidx] = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	jao    = jao0;
        /*
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        */
        BA[0] = LDG(atom_x[jat]) - (A[0] = LDG(atom_x[iat]));
        BA[1] = LDG(atom_y[jat]) - (A[1] = LDG(atom_y[iat]));
        BA[2] = LDG(atom_z[jat]) - (A[2] = LDG(atom_z[iat]));

        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        klcs0a = klcs0;
//	for (klcs = klcs0a ; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (klcs = klcs0a+tidx ; klcs<klcs1; klcs+=nthb ) {
#ifdef DLB_KL_PSSS
  NOT supported combination of defines.
#endif
#else
#ifndef DLB_KL_PSSS
//	for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
	for (int iklcs2 = tidx ; iklcs2<nklcs; iklcs2+=nthb ) {
          int iklcs;
          int it = iklcs2 / nthb;
          if (it%2 == 0) iklcs = iklcs2;
          else {
            int ilast = MIN2(nklcs, (it+1)*nthb);
            iklcs = it*nthb + ilast - iklcs2 - 1;
          }
#else
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
        while(iklcs<nklcs) {
#endif /* DLB_KL */
#endif /* USE_INSTANT_SCHWARZ */
            double DC[3], AC[3];
            float dmax;
            int klcs2;
#ifdef USE_INSTANT_SCHWARZ
	    float val_abcd = val_ab * csp_schwarz[klcs];
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
#else
            klcs = sklcs[iklcs];
#ifdef SORT_KL_SCHWARZ
            klcs = sorted_csp[klcs];
#endif
#endif /* USE_INSTANT_SCHWARZ */
            klcs2 = klcs - klcs0;
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao    = LDG(shel_ini[kcs]);
	    lao    = LDG(shel_ini[lcs]);
            /*
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
            */
            {
            double dtmp;
            AC[0] = A[0] - (dtmp = LDG(atom_x[kat]));
            DC[0] = LDG(atom_x[lat]) - dtmp;
            AC[1] = A[1] - (dtmp = LDG(atom_y[kat]));
            DC[1] = LDG(atom_y[lat]) - dtmp;
            AC[2] = A[2] - (dtmp = LDG(atom_z[kat]));
            DC[2] = LDG(atom_z[lat]) - dtmp;
            }
	    gpu_twoint_core_psss_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      PSSS );
            {
            double Gjk=ZERO, Gjl=ZERO, Glk2=ZERO;
	    double coe = ( kao == lao ? HALF : ONE );
            int j0 = jao*nao;
	    int jk = j0 + kao;
	    int jl = j0 + lao;
            int kl = kao*nao + lao;
#pragma unroll
	    for (int i=0, iao=iao0; i<3; i++, iao++ ) {
		    double x, x4;
//                    x = PSSS[i];
		if ( fabs(PSSS[i]) > eps_eri ) {
//		if ( fabs(x) > eps_eri ) {
//		    int ij, ik, il, jk, jl, kl, i0, j0;
		    int ij, ik, il, i0;
                    double dx;
		    x  = coe * PSSS[i];
                    x4 = 4.0 * x;
                    x *= -x_coef;
		    i0 = iao*nao;
		    ij = i0 + jao;
		    ik = i0 + kao;
		    il = i0 + lao;
		    i0 = i*nao;
                    /*
		    G[ij] += x4*Ds[kl];
		    G[kl] += x4*Ds[ij];
		    G[ik] -=  x*Ds[jl];
		    G[il] -=  x*Ds[jk];
		    G[jk] -=  x*Ds[il];
		    G[jl] -=  x*Ds[ik];
                    */
                    Gjk += x*LDG(Ds[il]);
                    Gjl += x*LDG(Ds[ik]);
		    Glk2 += x4*LDG(Ds[ij]);
		    Gij[i] += x4*LDG(Ds[kl]);
#ifdef USE_ATOMIC
                    dx = x*LDG(Ds[jl]);
		    atomicAdd(&G[ik],dx);
                    dx = x*LDG(Ds[jk]);
		    atomicAdd(&G[il],dx);
#else
		    Gi[i0+kao]   +=  x*LDG(Ds[jl]);
		    Gi[i0+lao]   +=  x*LDG(Ds[jk]);
#endif /* USE_ATOMIC */
		}
	    }	// for ( i, iao);
#ifdef USE_ATOMIC
		atomicAdd(&G[jk],Gjk);
		atomicAdd(&G[jl],Gjl);
		atomicAdd(&G[kl],Glk2);
#else
		Gj[kao] +=  Gjk;
		Gj[lao] +=  Gjl;
                Gkl[klcs2] += Glk2;
#endif
            }
#ifdef DLB_KL_PSSS
          if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
          iklcs = klcsw[widx] + tidw;
          }; // while (iklcs)
#endif
	}	// for ( klcs );
#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
#if 0
#pragma unroll
	  for (int i=0; i<3; i++) {
            int ij;
	    ij = jao0*nao + iao0 + i;
//            atomicAdd(&G[ij], sGij[i][tidx]);
            //atomicAdd(&G[ij], Gij[i]);
            atomicAdd(&G[ij], Gij[i]);
          }
#else
        {
          int i, j;
	  int i0 = iao0*nao;
	  int j0 = jao0*nao;
          double tGij[3];
          for (i=0; i<3; i++) tGij[i] = Gij[i];
          for (i=0; i<3; i++) {
            double *ss = &shared[sOffset+tidx];
            __syncthreads();
            *ss = tGij[i];
            { // Reduce Gij within warp
              double *sGij = &shared[sOffset+widx*WARP_SIZE];
              warpReduce(sGij, tidw);
            }
            __syncthreads();
            if (tidx==0) { // Reduce Gij for warp masters
              double *sGij = &shared[sOffset];
              double dtmp = sGij[0];
              for (j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
              atomicAdd(&G[j0+iao0+i], dtmp);
            }
          } // i
        }
#endif
        __syncthreads();
#else
//	klcs = klcs0;
        {
          int i,j;
          int nGi;
          int bs;
	  int i0 = iao0*nao;
	  int j0 = jao0*nao;
          int ijcs_next;
          int ics_next, jcs_next;
    /*
//	  for (i=0; i<3; i++) atomicAdd(&G[j0+iao0+i], Gij[i]);
//	  for (i=0; i<3; i++) G[j0+iao0+i] += Gij[i];
//            G[i0+i] += Gi[i];
//            G[j0+i] += Gj[i];
        */
//	  for (i=0; i<3; i++) Gj[iao0+i] += Gij[i];
//	  for (i=0; i<3; i++) Gi[jao0+nao*i] += Gij[i];
// Don't use Gj[iao0+i] because only s-type ao will be added for (ps,ss)
#pragma unroll
	  //for (i=0; i<3; i++) Gi[jao0+nao*i] += Gij[i];
	  for (i=0; i<3; i++) Gi[jao0+nao*i] += Gij[i];
//	  for (i=0; i<3; i++) Gi[jao0+nao*i] += sGij[i][tidx];
    __syncthreads();
          ics_next=jcs_next=-1;
#ifdef GPU_DLB
//          ijcs_next = ijcsw;
//          if (ijcs_next<ijcs1) { 
          ijcs_next = ijcs0 + ijcs1 - ijcsw - 1;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
//          if (ijcs_next>=ijcs0) { 
          if (ijcsw<ijcs1) { 
#else
          ijcs_next = ijcs2+nwkblk;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
          if (ijcs_next<ijcs1) { 
#endif
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
          /*
	  for (i=0; i<nao*3; i++) atomicAdd(&G[i0+i], Gi[i]);
	  for (i=0; i<nao; i++) atomicAdd(&G[j0+i], Gj[i]);
          */
          nGi = num_Gi*2;
          bs = bidx*nthb*num_Gi;
          if (ics != ics_next) {
          for (int ii=tidx;ii<numao*3;ii+=nthb) {
            i = (ii/numao)*nao + ii%numao + minao;
            {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
            /*
            for (j=0;j<nthb;j++) {
              Gtmp += *pG;
              *pG = ZERO;
              pG += num_Gi;
            }
            */
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
//            G[i0+i+ii*nao] += Gtmp;
            }
          } 
          }
//          }
          if (jcs != jcs_next) {
//          for (i=tidx;i<nao;i+=nthb) {
          for (i=minao+tidx;i<=maxao;i+=nthb) {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao*3 + i;
            double *pG1 = pG + num_Gi;
            /*
            for (j=0;j<nthb;j++) {
              Gtmp += *pG;
              *pG = ZERO;
              pG += num_Gi;
            }
            */
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          } 
          }
    __syncthreads();
//	  for ( i=0; i<nao*3; i++ ) Gi[i]=ZERO;
//	  for ( i=0; i<nao; i++ ) Gj[i]=ZERO;
        }
#endif /* USE_ATOMIC */
    }		// for ( ijcs );
    __syncthreads();
#ifndef USE_ATOMIC
    for (klcs = klcs0+tidx ; klcs<klcs1; klcs+=nthb ) {
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs];
      lao = shel_ini[lcs];
//      G[kao*nao+lao] += Gkl[klcs];
      G[kao*nao+lao] += Gkl[klcs-klcs0];
    }
#endif
    __syncthreads();
//    free(Gi);

    return;
}

/** (ps,ps)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
NREGS_128
gpu_twoint_direct_psps_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=1, Lb=0, Lc=1, Ld=0;
//    int Lab, Lcd, i, k, ix;
    int Lab, Lcd, i,j;
    int Labcd = 2;
    int *p_ijcounter = &ijcounter[Labcd];
//    int ipat, IJ, KL;
    int ijcs, ijcs0, ijcs1;
//    int klcs, klcs0, klcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
//    int ics, iat, iao, iao0, jcs, jat, jao;
    int ics, iat, iao0, jcs, jat, jao;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    double PSPS[3*3];
//    __shared__ double *PSPS;
    /*
    // for check sum
    double lsum = ZERO;
    */
    int klcs0a;
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
    __shared__ double BA[3];
//    __shared__ double A[3];
//    __shared__ double sGij[3][NTHREADS];
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[2];
    double *tbl = FMT_m_table2;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    double Gij[3];
    //double *Gij = &shared[tsize+1+tidx*3];
    size_t sOffset = tsize;
    tsize += nthb;
#else
    double *Gij = &shared[tidx*3];
    size_t tsize = nthb*3;
    size_t sOffset = 0;
#endif
#ifdef DLB_KL_PSPS
//    __shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
//    volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif

//    PSPS = (double *)malloc(sizeof(double)*3*3*nthb);
    mincs = MIN2(leading_cs[Lc], leading_cs[Ld]);
    minao = shel_ini[mincs];
    maxcs = MAX2(leading_cs[Lc+1], leading_cs[Ld+1]) - 1;
    maxao = shel_ini[maxcs+1]-1;
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
    Gkl = Gkl_b + bidx * num_Gkl;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao*3;
    work = work_b + bidx * nthb * WORK_SIZE;

    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lab];
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];

#ifndef USE_ATOMIC
    /*
    for ( i=0; i<nao*3; i++ ) Gi[i]=ZERO;
    for ( i=0; i<nao; i++ ) Gj[i]=ZERO;
    */
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*4; i+=nthb ) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0)*3; i+=nthb ) Gkl[i]=ZERO;
    __syncthreads();
#endif

#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
//    ijcsw = ijcs0 + iwk + nwks * bidx;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx;
      nijcsw = 0;
    }
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (ijcs=ijcs0+iwkblk; ijcs<ijcs1; ijcs+=nwkblk ) {
#endif
//        double A[3], BA[3];
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs1 - 1 - (ijcsw - ijcs0);
#endif
#ifdef DLB_KL_PSPS
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
        ics    = csp_ics[ijcs];
        jcs    = csp_jcs[ijcs];
        klcs0a = klcs0;
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else
        {
          float val_ab = csp_schwarz[ijcs];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
          for (int klcs = klcs0a+tidx ; klcs<=ijcs; klcs+=nthb) {
//	  for (int klcs = ijcs-tidx; klcs>=klcs0a ; klcs-=nthb) {
              int kcs, lcs;
              int iklcs;
              float dmax;
              float val_abcd = val_ab * csp_schwarz[klcs];
              if ( val_abcd < eps_ps4 ) continue;
              kcs    = csp_ics[klcs];
              lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
              if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif
#pragma unroll
        for (int i=0;i<3;i++) Gij[i] = ZERO;
//        for (i=0;i<3;i++) sGij[i][tidx] = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao    = shel_ini[jcs];
        {
          /*
        double B[3];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        */
        /*
	BA[0] =atom_x[jat] - (A[0]=atom_x[iat]);
	BA[1] =atom_y[jat] - (A[1]=atom_y[iat]);
	BA[2] =atom_z[jat] - (A[2]=atom_z[iat]);
        */
        /*
	BA[0] =atom_x[jat] - (work[tidx       ]=atom_x[iat]);
	BA[1] =atom_y[jat] - (work[tidx+nthb  ]=atom_y[iat]);
	BA[2] =atom_z[jat] - (work[tidx+nthb*2]=atom_z[iat]);
        */
	  BA[0] = LDG(atom_x[jat]) - (WORK(0) = LDG(atom_x[iat]));
	  BA[1] = LDG(atom_y[jat]) - (WORK(1) = LDG(atom_y[iat]));
	  BA[2] = LDG(atom_z[jat]) - (WORK(2) = LDG(atom_z[iat]));
        }

	
//	for ( ; klcs<=ijcs; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs = klcs0a+tidx ; klcs<=ijcs; klcs+=nthb ) {
#else
#ifndef DLB_KL_PSPS
        for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
#else
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
        while(iklcs<nklcs) {
#endif /* DLB_KL */
#endif
            int klcs2;
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
            int ipat;
            int ijgekl;
            double DC[3], AC[3];
#ifdef USE_INSTANT_SCHWARZ
            {
	    float val_abcd = val_ab * csp_schwarz[klcs];
            float dmax;
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
            }
#else
            int klcs;
            klcs = sklcs[iklcs];
#endif
            klcs2 = (klcs - klcs0)*3;
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao0   = LDG(shel_ini[kcs]);
	    lao    = LDG(shel_ini[lcs]);
            /*
            {
              double C[3],D[3];
	    C[0]=LDG(atom_x[kat]); C[1]=LDG(atom_y[kat]); C[2]=LDG(atom_z[kat]);
	    D[0]=LDG(atom_x[lat]); D[1]=LDG(atom_y[lat]); D[2]=LDG(atom_z[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
//		AC[i] = A[i] - C[i];
//		AC[i] = work[tidx+nthb*i] - C[i];
		AC[i] = WORK(i) - C[i];
		DC[i] = D[i] - C[i];
	    }
            }
            */
            {
            double dtmp;
            AC[0] = WORK(0) - (dtmp = LDG(atom_x[kat]));
            DC[0] = LDG(atom_x[lat]) - dtmp;
            AC[1] = WORK(1) - (dtmp = LDG(atom_y[kat]));
            DC[1] = LDG(atom_y[lat]) - dtmp;
            AC[2] = WORK(2) - (dtmp = LDG(atom_z[kat]));
            DC[2] = LDG(atom_z[lat]) - dtmp;
            }
	    gpu_twoint_core_psps_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      PSPS );
	    ipat = ( (ics==kcs && jcs>lcs) ? true : false);
            ijgekl = (ics>kcs);
            if (ics==kcs) ijgekl = (jcs>=lcs);
            if (!ijgekl) ipat = ( (ics==kcs && jcs<lcs) ? true : false);
            {
            int iao, ix;
            int j0 = jao*nao;
            int jl = j0 + lao;
            double Gjl = ZERO;
            double Gjk[3]={ZERO,ZERO,ZERO};
            double Glk2[3]={ZERO,ZERO,ZERO};
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<3; i++, iao++ ) {
                int k, kao;
		int i0 = iao*nao;
		int il = i0 + lao;
                double Gil = ZERO;
		int IJ = ((iao*iao+iao)>>1) + jao;
#pragma unroll
		for ( k=0, kao=kao0; k<3; k++, kao++, ix++ ) {
		    int KL = ((kao*kao+kao)>>1) + lao ;
		    if ( fabs(PSPS[ix]) <= eps_eri ) continue;
//		    if ( IJ>=KL || ipat ) {
		    if ( (ijgekl&&IJ>=KL) || (!ijgekl&&KL>=IJ) || ipat ) {
			double x, x4, coe;
                        double dx;
//			int ij, ik, il, jk, jl, kl, i0, j0;
			int ij, ik, jk, kl;
			coe = ONE;
			if ( IJ == KL ) coe = HALF;
			x  = coe * PSPS[ix];
			x4 = 4.e0 * x;
                        x *= -x_coef;
//			i0 = iao*nao;
//			j0 = jao*nao;
			ij = i0 + jao;
			ik = i0 + kao;
//			il = i0 + lao;
			jk = j0 + kao;
//			jl = j0 + lao;
			kl = kao*nao + lao;
/*
			G[ij] += x4*Ds[kl];
			G[kl] += x4*Ds[ij];
			G[ik] -=  x*Ds[jl];
			G[il] -=  x*Ds[jk];
			G[jk] -=  x*Ds[il];
			G[jl] -=  x*Ds[ik];
*/
#ifdef USE_ATOMIC
                        dx = x*LDG(Ds[jl]);
                        atomicAdd(&G[ik],dx);
#else
                        Gi[i*nao+kao] += x*LDG(Ds[jl]);
#endif
                        Gil    += x*LDG(Ds[jk]);
                        Gjk[k] += x*LDG(Ds[il]);
                        Gjl    += x*LDG(Ds[ik]);
                        Gij[i]  += x4*LDG(Ds[kl]);
                        Glk2[k] += x4*LDG(Ds[ij]);
		    }
		}	// for ( kao )
#ifdef USE_ATOMIC
              atomicAdd(&G[il],Gil);
#else
              Gi[i*nao+lao] += Gil;
#endif
	    }		// for ( iao )
#ifdef USE_ATOMIC
              atomicAdd(&G[jl],Gjl);
#pragma unroll
              for (int i=0;i<3;i++) atomicAdd(&G[j0+kao0+i],Gjk[i]);
#pragma unroll
              for (int k=0;k<3;k++) atomicAdd(&G[lao*nao+kao0+k],Glk2[k]);
#else
              Gj[lao] += Gjl;
#pragma unroll
              for (int i=0;i<3;i++) Gj[kao0+i] += Gjk[i];
#pragma unroll
              for (int k=0;k<3;k++) Gkl[klcs2+k] += Glk2[k];
#endif
            }
#ifdef DLB_KL_PSPS
          if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
          iklcs = klcsw[widx] + tidw;
          }; // while (iklcs)
#endif
	}	// for ( klcs );
#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
#if 0
#pragma unroll
	  for (int i=0; i<3; i++) {
            int ij;
	    ij = jao*nao + iao0 + i;
//            atomicAdd(&G[ij], sGij[i][tidx]);
            atomicAdd(&G[ij], Gij[i]);
          }
#else
        {
#ifndef CUDA_FMT_M_SM
          double tGij[3];
          for (int i=0; i<3; i++) tGij[i] = Gij[i];
#else
          double *tGij = Gij;
#endif
          for (int i=0; i<3; i++) {
            double *ss = &shared[sOffset+tidx];
            __syncthreads();
            *ss = tGij[i];
            { // Reduce Gij within warp
              double *sGij = &shared[sOffset+widx*WARP_SIZE];
              warpReduce(sGij, tidw);
            }
            __syncthreads();
            if (tidx==0) { // Reduce Gij for warp masters
              double *sGij = &shared[sOffset];
              double dtmp = sGij[0];
              for (int j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
              atomicAdd(&G[jao*nao+iao0+i], dtmp);
            }
          } // i
        }
#endif
    __syncthreads();
#else
//	klcs = klcs0;
//        if (n2ei>0) {
        {
          int i, j;
          int nGi;
          int bs;
	  int i0 = iao0*nao;
	  int j0 = jao*nao;
          int ijcs_next;
          int ics_next=-1, jcs_next=-1;
#pragma unroll
	  for (i=0; i<3; i++) Gj[iao0+i] += Gij[i];
//	  for (i=0; i<3; i++) Gj[iao0+i] += sGij[i][tidx];
//	  for (i=0; i<3; i++) Gi[jao+nao*i] += sGij[i][tidx];
    __syncthreads();
#ifdef GPU_DLB
//          ijcs_next = ijcsw;
          ijcs_next = ijcs1 - ijcsw + 1;
          ijcs_next = ijcs1 - 1 - (ijcsw - ijcs0);
          if (ijcsw<ijcs1) {
#else
          ijcs_next = ijcs+nwkblk;
          if (ijcs_next<ijcs1) {
#endif
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
          nGi = num_Gi*2;
          bs = bidx*nthb*num_Gi;
          if (ics != ics_next) {
//          for (int ii=tidx;ii<nao*3;ii+=nthb) {
//            i = (ii/nao)*nao + ii%nao;
          for (int ii=tidx;ii<numao*3;ii+=nthb) {
            i = (ii/numao)*nao + ii%numao + minao;
            {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
            /*
            for (j=0;j<nthb;j++) {
              Gtmp += *pG;
              *pG = ZERO;
              pG += num_Gi;
            }
            */
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
            }
          } 
          }
          if (jcs != jcs_next) {
//          for (i=tidx;i<nao;i+=nthb) {
          for (i=minao+tidx;i<=maxao;i+=nthb) {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao*3 + i;
            double *pG1 = pG + num_Gi;
            /*
            for (j=0;j<nthb;j++) {
              Gtmp += *pG;
              *pG = ZERO;
              pG += num_Gi;
            }
            */
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          } 
          }
    __syncthreads();
//	  for ( i=0; i<nao*3; i++ ) Gi[i]=ZERO;
//	  for ( i=0; i<nao; i++ ) Gj[i]=ZERO;
        }
#endif /* USE_ATOMIC */
//	klcs = klcs0;
    }		// for ( ijcs );
#ifndef USE_ATOMIC
    /*
    for (int klcs = klcs0+tidx ; klcs<klcs1; klcs+=nthb ) {
      int klcs2;
      int kcs,lcs,kao,lao,l0;
      klcs = klcs0 + i/3;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs];
      lao = shel_ini[lcs];
      l0 = lao*nao+kao;
#pragma unroll
      for (i=0;i<3;i++) G[l0+i] += Gkl[klcs2+i];
    }
    */
    for (int i=tidx; i<(klcs1-klcs0)*3; i+=nthb) {
      int klcs, k;
      int kcs,lcs,kao,lao,l0;
      klcs = i/3;
      k = i - klcs*3;
      klcs += klcs0;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs];
      lao = shel_ini[lcs];
      l0 = lao*nao+kao;
      G[l0+k] += Gkl[i];
    }
#endif

    return;
}


/** (pp,ss)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
// exchange La,Lb <=> Lc,Ld
__global__ void
NREGS_128
gpu_twoint_direct_ppss_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=1, Lb=1, Lc=0, Ld=0;
    int Lab, Lcd;
    int Labcd = 3;
    int *p_ijcounter = &ijcounter[Labcd];
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, klcs0a;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, lcs, lat, lao;
    double PPSS[3*3];
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
//    double coe;
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    int ics2, jcs2;
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[2];
    double *tbl = FMT_m_table2;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    //double Gij[1];
    size_t sOffset = tsize;
    double *Gij = &shared[sOffset+tidx];
    tsize += nthb;
#else
    double *Gij = &shared[tidx];
    size_t tsize = nthb*3;
    size_t sOffset = 0;
#endif
#ifdef DLB_KL_PPSS
//    __shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
//    volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif
    __shared__ double A[3];
    __shared__ double BA[3];

    mincs = MIN2(leading_cs[La], leading_cs[Lb]);
    minao = shel_ini[mincs];
    maxcs = MAX2(leading_cs[La+1], leading_cs[Lb+1]) - 1;
    maxao = shel_ini[maxcs+1]-1;
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
    Gkl = Gkl_b + bidx * num_Gkl;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao;
    work = work_b + bidx * nthb * WORK_SIZE;

//    nwkblk = nworkers * ndev * nblk;
//    iwkblk = workerid * ndev * nblk + idev * nblk + bidx;
//    nwkblk = nwks;
//    iwkblk = iwk + bidx;
    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;

    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lcd];
    ijcs1 = leading_cs_pair[Lcd+1];
    klcs0 = leading_cs_pair[Lab];
    klcs1 = leading_cs_pair[Lab+1];

#ifndef USE_ATOMIC
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*2; i+=nthb ) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0)*3*3; i+=nthb) Gkl[i]=ZERO;
    __syncthreads();
#endif

#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
//    ijcsw = ijcs0 + iwk + nwks * bidx;
//    nijcsw = 0;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx * NIJCSW;
      nijcsw = NIJCSW-1;
    }
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (int ijcs2=ijcs0+iwkblk; ijcs2<ijcs1; ijcs2+=nwkblk ) {
#endif
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs0 + ijcs1 - ijcsw - 1;
        /*
        __syncthreads();
        if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
        */
#else
      ijcs = ijcs2;
#endif
#ifdef SORT_IJ_SCHWARZ
      ijcs = sorted_csp[ijcs];
#endif
#ifdef DLB_KL_PPSS
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
//        if(tidx==0) printf("%d:0 %d %d %d\n",tidx,ijcs,ics,jcs);
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else /* !USE_INSTANT_SCHWARZ */
        {
	  float val_ab = csp_schwarz[ijcs];
//          int sklcs_l[NKLBUF];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
	  for (int klcs2 = klcs0+tidx ; klcs2<klcs1; klcs2+=nthb) {
//	  for (int klcs2 = klcs1-tidx-1; klcs2>=klcs0 ; klcs2-=nthb) {
              int klcs=klcs2;
              int kcs, lcs;
              int iklcs;
              float dmax;
	      float val_abcd;
#ifdef SORT_KL_SCHWARZ
          klcs = sorted_csp[klcs2];
#endif
	      val_abcd = val_ab * csp_schwarz[klcs];
	      if ( val_abcd < eps_ps4 ) continue;
	      kcs    = csp_ics[klcs];
	      lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	      if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
#ifdef SORT_KL_SCHWARZ
              klcs = klcs2;
#endif
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
//          if(bidx==0&&tidx==0) printf("nklcs = %d\n", nklcs);
        }
#endif /* USE_INSTANT_SCHWARZ */
        *Gij  = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao    = shel_ini[ics];
	jao    = shel_ini[jcs];
        __syncthreads();
        if(tidx == 0) {
        /*
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        */
        BA[0] = LDG(atom_x[jat]) + (A[0] = -LDG(atom_x[iat]));
        BA[1] = LDG(atom_y[jat]) + (A[1] = -LDG(atom_y[iat]));
        BA[2] = LDG(atom_z[jat]) + (A[2] = -LDG(atom_z[iat]));
        /*
	  BA[0] = atom_x[jat] + (WORK(0) = -atom_x[iat]);
	  BA[1] = atom_y[jat] + (WORK(1) = -atom_y[iat]);
	  BA[2] = atom_z[jat] + (WORK(2) = -atom_z[iat]);
        */
        }
        __threadfence_block();
        __syncthreads();

//        if(tidx==0) printf("%d:k %d %d\n",tidx,klcs0,klcs1);
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
//	for (klcs = klcs0 ; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (klcs = klcs0+tidx ; klcs<klcs1; klcs+=nthb ) {
#ifdef DLB_KL_PPSS
  NOT supported combination of defines.
#endif
#else /* !USE_INSTANT_SCHWARZ */
#ifndef DLB_KL_PPSS
//	for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
	for (int iklcs2 = tidx ; iklcs2<nklcs; iklcs2+=nthb ) {
          int iklcs;
          int it = iklcs2 / nthb;
          if (it%2 == 0) iklcs = iklcs2;
          else {
            int ilast = MIN2(nklcs, (it+1)*nthb);
            iklcs = it*nthb + ilast - iklcs2 - 1;
          }
#else /* !DLB_KL_PPSS */
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
        while(iklcs<nklcs) {
#endif /* !DLB_KL_PPSS */
#endif /* !USE_INSTANT_SCHWARZ */
            double DC[3], AC[3];
            float dmax;
            int jk, jl, j0;
            int kao0, lao0, kao, lao;
//            int n2ei2 = 0;
            int klcs2;
#ifdef USE_INSTANT_SCHWARZ
	    float val_abcd = val_ab * csp_schwarz[klcs];
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
#else
            klcs = sklcs[iklcs];
#ifdef SORT_KL_SCHWARZ
            klcs = sorted_csp[klcs];
#endif
#endif /* USE_INSTANT_SCHWARZ */
            klcs2 = klcs - klcs0;
//         if (tidx==0) printf("%d:kl %d %d\n",tidx,klcs, klcs2);
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
//            Gjk = Gjl = ZERO;
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao0   = LDG(shel_ini[kcs]);
	    lao0   = LDG(shel_ini[lcs]);
            /*
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
            */
            {
              double dtmp;
              AC[0] = A[0] + (dtmp = LDG(atom_x[kat]));
              DC[0] = LDG(atom_x[lat]) - dtmp;
              AC[1] = A[1] + (dtmp = LDG(atom_y[kat]));
              DC[1] = LDG(atom_y[lat]) - dtmp;
              AC[2] = A[2] + (dtmp = LDG(atom_z[kat]));
              DC[2] = LDG(atom_z[lat]) - dtmp;
              /*
              AC[0] =  WORK(0) + (dtmp = atom_x[kat]);
              DC[0] = atom_x[lat] - dtmp;
              AC[1] =  WORK(1) + (dtmp = atom_y[kat]);
              DC[1] = atom_y[lat] - dtmp;
              AC[2] =  WORK(2) + (dtmp = atom_z[kat]);
              DC[2] = atom_z[lat] - dtmp;
              */
            }
/*
	    gpu_twoint_core_ppss_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      PPSS );
*/
// (ss,pp) -> (pp,ss)
	    gpu_twoint_core_ppss_(
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,   AC,      PPSS );

//        printf("%d:core %d %d\n",tidx,nijps, nklps);
            {
              int i0=iao*nao, j0=jao*nao;
              int ij = i0 + jao;
              double Gil[3] = {ZERO, ZERO, ZERO};
              double Gjl[3] = {ZERO, ZERO, ZERO};
            double coe, coe0;
	    coe0 = ( iao == jao ? HALF : ONE );
#pragma unroll
	    for ( int k=0, kao=kao0, ix=0; k<3; k++, kao++ ) {
              int ik = i0 + kao;
              int jk = j0 + kao;
              double Gik = ZERO, Gjk = ZERO;
#pragma unroll
		for ( int l=0, lao=lao0; l<3; l++, lao++, ix++ ) {
		    if ( lao > kao ) continue;
		    if ( fabs(PPSS[ix]) > eps_eri ) {
			double x, x4;
                        double dx;
			int il, jl, kl;
			coe = coe0;
			if ( kao == lao ) coe *= HALF;
			x  = coe * PPSS[ix];

			x4 = 4.e0 * x;
                        x *= -x_coef;
			il = i0 + lao;
			jl = j0 + lao;
			kl = kao*nao + lao;
                        /*
			G[ij] += x4*Ds[kl];
			G[kl] += x4*Ds[ij];
			G[ik] -=  x*Ds[jl];
			G[il] -=  x*Ds[jk];
			G[jk] -=  x*Ds[il];
			G[jl] -=  x*Ds[ik];
                        */

			Gil[l] += x*LDG(Ds[jk]);
			Gjl[l] += x*LDG(Ds[ik]);
			Gik += x*LDG(Ds[jl]);
			Gjk += x*LDG(Ds[il]);
                        *Gij += x4*LDG(Ds[kl]);
#ifdef USE_ATOMIC
			dx = x4*LDG(Ds[ij]);
                        atomicAdd(&G[kl], dx);
#else
                        Gkl[klcs2*9+ix] += x4*LDG(Ds[ij]);
#endif
		    }
		} // for ( lao );
#ifdef USE_ATOMIC
                atomicAdd(&G[ik], Gik);
                atomicAdd(&G[jk], Gjk);
#else
                Gi[kao] += Gik;
                Gj[kao] += Gjk;
#endif
	    }	// for ( kao );
#pragma unroll
            for ( int l=0, lao=lao0; l<3; l++, lao++ ) {
              int il = i0 + lao;
              int jl = j0 + lao;
#ifdef USE_ATOMIC
              atomicAdd(&G[il], Gil[l]);
              atomicAdd(&G[jl], Gjl[l]);
#else
              Gi[lao] += Gil[l];
              Gj[lao] += Gjl[l];
#endif
            }
            }  // block for additions
#ifdef DLB_KL_PPSS
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx] + tidw;
          } // while (iklcs)
#endif
	}	// for ( klcs );
#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
#if 0
        atomicAdd(&G[iao*nao+jao], *Gij);
#else
          { // Reduce Gij within warp
            //double *sGij = &shared[widx*WARP_SIZE];
            double *sGij = &shared[sOffset+widx*WARP_SIZE];
            warpReduce(sGij, tidw);
          }
          __syncthreads();
          if (tidx==0) { // Reduce Gij for warp masters
            //double *sGij = &shared[0];
            double *sGij = &shared[sOffset];
            double dtmp = sGij[0];
            for (int j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
            atomicAdd(&G[iao*nao+jao], dtmp);
          }
#endif
    __syncthreads();
#else
//	klcs = klcs0;
//        if (n2ei>0) {
        {
          int i,j;
          int nGi;
          int bs;
	  int i0 = iao*nao;
	  int j0 = jao*nao;
          int ijcs_next;
          int ics_next, jcs_next;
#if 1
          { // Reduce Gij within warp
            //double *sGij = &shared[widx*WARP_SIZE];
            double *sGij = &shared[sOffset+widx*WARP_SIZE];
            warpReduce(sGij, tidw);
          }
          __syncthreads();
          if (tidx==0) { // Reduce Gij for warp masters
            //double *sGij = &shared[0];
            double *sGij = &shared[sOffset];
            double dtmp = sGij[0];
            for (j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
            G[i0+jao] += dtmp;
          }
#else
        atomicAdd(&G[iao*nao+jao], *Gij);
        //Gi[jao] += *Gij;
#endif
          ics_next=jcs_next=-1;
#ifdef GPU_DLB
          __syncthreads();
//          ijcs_next = ijcsw;
//          if (ijcs_next<ijcs1) { 
          ijcs_next = ijcs0 + ijcs1 - ijcsw - 1;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
//          if (ijcs_next>=ijcs0) { 
          if (ijcsw<ijcs1) { 
#else /* !GPU_DLB */
          ijcs_next = ijcs2+nwkblk;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
          if (ijcs_next<ijcs1) { 
#endif /* !GPU_DLB */
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
#ifndef GPU_DLB
          if (ics != ics_next || jcs != jcs_next) __syncthreads();
#endif
          /*
	  for (i=0; i<nao*3; i++) atomicAdd(&G[i0+i], Gi[i]);
	  for (i=0; i<nao; i++) atomicAdd(&G[j0+i], Gj[i]);
          */
          nGi = num_Gi*2;
          bs = bidx*nthb*num_Gi;
          if (ics != ics_next) {
          for (int ii=tidx;ii<numao;ii+=nthb) {
            i = (ii/numao)*nao + ii%numao + minao;
            {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
            }
          } // for ii 
          } // ics != ics_next
          if (jcs != jcs_next) {
//          for (i=tidx;i<nao;i+=nthb) {
          for (i=minao+tidx;i<=maxao;i+=nthb) {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          } // for i
          } // jcs != jcs_next
    __syncthreads();
        }
#endif /* !USE_ATOMIC */
//	klcs = klcs0;
    }		// for ( ijcs );

#ifndef USE_ATOMIC
    for (int i=tidx; i<(klcs1-klcs0)*3*3; i+=nthb) {
      int klcs, kl;
      int kcs,lcs,kao,lao;
      int kao0, lao0, k, l;
      klcs = i/9;
      kl = i - klcs*9;
      k = kl/3;
      l = kl - k*3;
      klcs += klcs0;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs] + k;
      lao = shel_ini[lcs] + l;
      G[kao*nao+lao] += Gkl[i];
    }
#endif

    return;
}

/** (pp,ps)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
NREGS_128
gpu_twoint_direct_ppps_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=1, Lb=1, Lc=1, Ld=0;
    int Lab, Lcd;
    int Labcd = 4;
    int *p_ijcounter = &ijcounter[Labcd];
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, klcs0a;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao0, jcs, jat, jao0;
    int kcs, kat, kao0, lcs, lat, lao0;
    double PPPS[3*3*3];
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    int ics2, jcs2;
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[3];
    double *tbl = FMT_m_table3;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    double Gij[3];
    //double *Gij = &shared[tsize+1+tidx*3];
    size_t sOffset = tsize;
    tsize += nthb;
#else
    double *Gij = &shared[tidx*3];
    size_t tsize = nthb*3;
    size_t sOffset = 0;
#endif
#ifdef DLB_KL_PPPS
//    __shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
//    volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif
    __shared__ double A[3];
    __shared__ double BA[3];

    mincs = MIN2(leading_cs[La], leading_cs[Lb]);
    minao = shel_ini[mincs];
    maxcs = MAX2(leading_cs[La+1], leading_cs[Lb+1]) - 1;
    maxao = shel_ini[maxcs+1]-1;
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
    Gkl = Gkl_b + bidx * num_Gkl;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao*3;
    work = work_b + bidx * nthb * WORK_SIZE;

//    nwkblk = nworkers * ndev * nblk;
//    iwkblk = workerid * ndev * nblk + idev * nblk + bidx;
//    nwkblk = nwks;
//    iwkblk = iwk + bidx;
    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;

    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lcd];
    ijcs1 = leading_cs_pair[Lcd+1];
    klcs0 = leading_cs_pair[Lab];
    klcs1 = leading_cs_pair[Lab+1];

#ifndef USE_ATOMIC
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*4; i+=nthb ) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0)*3*3; i+=nthb) Gkl[i]=ZERO;
    __syncthreads();
#endif

#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
//    ijcsw = ijcs0 + iwk + nwks * bidx;
//    nijcsw = 0;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx * NIJCSW;
      nijcsw = NIJCSW-1;
    }
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (int ijcs2=ijcs0+iwkblk; ijcs2<ijcs1; ijcs2+=nwkblk ) {
#endif
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs0 + ijcs1 - ijcsw - 1;
        /*
        __syncthreads();
        if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
        */
#else
      ijcs = ijcs2;
#endif
#ifdef SORT_IJ_SCHWARZ
      ijcs = sorted_csp[ijcs];
#endif
#ifdef DLB_KL_PPPS
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
//        if(tidx==0) printf("%d:0 %d %d %d\n",tidx,ijcs,ics,jcs);
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else /* !USE_INSTANT_SCHWARZ */
        {
	  float val_ab = csp_schwarz[ijcs];
//          int sklcs_l[NKLBUF];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
	  for (int klcs2 = klcs0+tidx ; klcs2<klcs1; klcs2+=nthb) {
//	  for (int klcs2 = klcs1-tidx-1; klcs2>=klcs0 ; klcs2-=nthb) {
              int klcs=klcs2;
              int kcs, lcs;
              int iklcs;
              float dmax;
	      float val_abcd;
#ifdef SORT_KL_SCHWARZ
          klcs = sorted_csp[klcs2];
#endif
	      val_abcd = val_ab * csp_schwarz[klcs];
	      if ( val_abcd < eps_ps4 ) continue;
	      kcs    = csp_ics[klcs];
	      lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	      if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
#ifdef SORT_KL_SCHWARZ
              klcs = klcs2;
#endif
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
//          if(bidx==0&&tidx==0) printf("nklcs = %d\n", nklcs);
        }
#endif /* USE_INSTANT_SCHWARZ */
#pragma unroll
        for (int i=0;i<3;i++) Gij[i] = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
        __syncthreads();
        if(tidx == 0) {
        /*
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        */
        BA[0] = LDG(atom_x[jat]) + (A[0] = -LDG(atom_x[iat]));
        BA[1] = LDG(atom_y[jat]) + (A[1] = -LDG(atom_y[iat]));
        BA[2] = LDG(atom_z[jat]) + (A[2] = -LDG(atom_z[iat]));
        }
        __threadfence_block();
        __syncthreads();

//        if(tidx==0) printf("%d:k %d %d\n",tidx,klcs0,klcs1);
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
//	for (klcs = klcs0 ; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (klcs = klcs0+tidx ; klcs<klcs1; klcs+=nthb ) {
#ifdef DLB_KL_PPPS
  NOT supported combination of defines.
#endif
#else /* !USE_INSTANT_SCHWARZ */
#ifndef DLB_KL_PPPS
//	for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
	for (int iklcs2 = tidx ; iklcs2<nklcs; iklcs2+=nthb ) {
          int iklcs;
          int it = iklcs2 / nthb;
          if (it%2 == 0) iklcs = iklcs2;
          else {
            int ilast = MIN2(nklcs, (it+1)*nthb);
            iklcs = it*nthb + ilast - iklcs2 - 1;
          }
#else /* !DLB_KL_PPPS */
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
        while(iklcs<nklcs) {
#endif /* !DLB_KL_PPPS */
#endif /* !USE_INSTANT_SCHWARZ */
            double DC[3], AC[3];
            float dmax;
//            int n2ei2 = 0;
            int klcs2;
#ifdef USE_INSTANT_SCHWARZ
	    float val_abcd = val_ab * csp_schwarz[klcs];
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
#else
            klcs = sklcs[iklcs];
#ifdef SORT_KL_SCHWARZ
            klcs = sorted_csp[klcs];
#endif
#endif /* USE_INSTANT_SCHWARZ */
            klcs2 = klcs - klcs0;
//         if (tidx==0) printf("%d:kl %d %d\n",tidx,klcs, klcs2);
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
//            Gjk = Gjl = ZERO;
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao0   = LDG(shel_ini[kcs]);
	    lao0   = LDG(shel_ini[lcs]);
            /*
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
            */
            { // AC -> CA
              double dtmp;
              AC[0] = A[0] + (dtmp = LDG(atom_x[kat]));
              DC[0] = LDG(atom_x[lat]) - dtmp;
              AC[1] = A[1] + (dtmp = LDG(atom_y[kat]));
              DC[1] = LDG(atom_y[lat]) - dtmp;
              AC[2] = A[2] + (dtmp = LDG(atom_z[kat]));
              DC[2] = LDG(atom_z[lat]) - dtmp;
            }
/*
	    gpu_twoint_core_ppps_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      PPPS );
*/
// (ps,pp) -> (pp,ps)
#ifdef GPU_TWOINT_RYS
            gpu_twoint_core_rys_ppps
#else
	    gpu_twoint_core_ppps_
#endif
		   (&nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,   AC,      PPPS);
          {
            double coe;
            int iao, jao = jao0;
//            int j0 = jao*nao;
            double Gjl[3] = {ZERO, ZERO, ZERO};
            double Gil[9];
#pragma unroll
            for (int i=0;i<9;i++) Gil[i]=ZERO;
#pragma unroll
	    for (int k=0, kao=kao0, ix=0; k<3; k++, kao++ ) {
//              int jk = j0 + kao;
              double Gjk = ZERO;
              double Gik[3] = {ZERO, ZERO, ZERO};
#pragma unroll
		for (int l=0, lao=lao0; l<3; l++, lao++ ) {
//                  int jl = j0 + lao;
                  double Gkl2 = ZERO;
//                  double Gjl = ZERO;
//                  int kl = kao*nao + lao;
		    if ( lao>kao ) { ix += 3; continue; }
		    coe = ( kao==lao ? HALF : ONE );
#pragma unroll
		    for (int i=0, iao=iao0; i<3; i++, iao++, ix++ ) {
                      double x = PPPS[ix];
//			if ( fabs(PPPS[ix]) > eps_eri ) {
			if ( fabs(x) > eps_eri ) {
                            double x4, dx;
			    x  *= coe;

			    x4 = 4.e0 * x;
                            x *= -x_coef;
                            /*
			    G[ij] += x4*Ds[kl];
			    G[kl] += x4*Ds[ij];
			    G[ik] -=  x*Ds[jl];
			    G[il] -=  x*Ds[jk];
			    G[jk] -=  x*Ds[il];
			    G[jl] -=  x*Ds[ik];
                            */
                        Gij[i] += x4*LDG(Ds[kao*nao+lao]);
                        Gkl2   += x4*LDG(Ds[jao0*nao+iao]);
                        Gjk    += x*LDG(Ds[lao*nao+iao]);
                        Gjl[l] += x*LDG(Ds[kao*nao+iao]);
                        Gik[i] += x*LDG(Ds[jao0*nao+lao]);
                        Gil[l*3+i] += x*LDG(Ds[jao0*nao+kao]);
			}	// if ( fabs );
		    }	// for ( iao )
#ifdef USE_ATOMIC
                  atomicAdd(&G[kao*nao+lao], Gkl2);
#else
                  Gkl[klcs2*9+k*3+l] += Gkl2;
#endif
		}	// for ( lao )
#ifdef USE_ATOMIC
              atomicAdd(&G[jao0*nao+kao], Gjk);
#pragma unroll
              for (int i=0,k0=kao*nao+iao0; i<3; i++) atomicAdd(&G[k0+i], Gik[i]);
#else
              Gj[kao] += Gjk;
#pragma unroll
              for (int i=0; i<3; i++) Gi[i*nao+kao] += Gik[i];
#endif
	    }	// for ( kao )
#ifdef USE_ATOMIC
#pragma unroll
            for (int l=0,jl=jao0*nao+lao0; l<3; l++,jl++)
              atomicAdd(&G[jl], Gjl[l]);
#pragma unroll
            for (int l=0,lao=lao0; l<3; l++,lao++) 
#pragma unroll
              for (int i=0,il=lao*nao+iao0; i<3; i++,il++)
                atomicAdd(&G[il], Gil[l*3+i]);
#else
#pragma unroll
            for (int l=0; l<3; l++) Gj[lao0+l] += Gjl[l];
#pragma unroll
            for (int l=0,lao=lao0; l<3; l++,lao++) 
#pragma unroll
              for (int i=0; i<3; i++)
                Gi[i*nao+lao] += Gil[l*3+i];
#endif
          } // block for additions
#ifdef DLB_KL_PPPS
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx] + tidw;
          } // while (iklcs)
#endif
	}	// for ( klcs );
#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
#if 0
#pragma unroll
      //for (i=0; i<3; i++) atomicAdd(&G[jao0*nao+iao0+i], Gij[i]);
      for (int i=0,ji=jao0*nao+iao0; i<3; i++) atomicAdd(&G[ji+i], Gij[i]);
#else
        {
#ifndef CUDA_FMT_M_SM
          double tGij[3];
          for (int i=0; i<3; i++) tGij[i] = Gij[i];
#else
          double *tGij = Gij;
#endif
          for (int i=0; i<3; i++) {
            double *ss = &shared[sOffset+tidx];
            __syncthreads();
            *ss = tGij[i];
            { // Reduce Gij within warp
              double *sGij = &shared[sOffset+widx*WARP_SIZE];
              warpReduce(sGij, tidw);
            }
            __syncthreads();
            if (tidx==0) { // Reduce Gij for warp masters
              double *sGij = &shared[sOffset];
              double dtmp = sGij[0];
              for (int j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
              atomicAdd(&G[jao0*nao+iao0+i], dtmp);
            }
          } // i
        }
#endif
      __syncthreads();
#else
//	klcs = klcs0;
//        if (n2ei>0) {
        {
          int i,j;
          int nGi;
          int bs;
	  int i0 = iao0*nao;
	  int j0 = jao0*nao;
          int ijcs_next;
          int ics_next, jcs_next;
          ics_next=jcs_next=-1;
#pragma unroll
          for (i=0; i<3; i++) Gj[iao0+i] += Gij[i];
#ifdef GPU_DLB
          __syncthreads();
//          ijcs_next = ijcsw;
//          if (ijcs_next<ijcs1) { 
          ijcs_next = ijcs0 + ijcs1 - ijcsw - 1;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
//          if (ijcs_next>=ijcs0) { 
          if (ijcsw<ijcs1) { 
#else /* !GPU_DLB */
          ijcs_next = ijcs2+nwkblk;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
          if (ijcs_next<ijcs1) { 
#endif /* !GPU_DLB */
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
#ifndef GPU_DLB
          if (ics != ics_next || jcs != jcs_next) __syncthreads();
#endif
          /*
          if (ics != ics_next) {
	  for (i=0; i<nao*3; i++) atomicAdd(&G[iao0*nao+i], Gi[i]);
	  for (i=0; i<nao*3; i++) Gi[i]=ZERO;
          }
          if (jcs != jcs_next) {
	  for (i=0; i<nao; i++) atomicAdd(&G[jao0*nao+i], Gj[i]);
	  for (i=0; i<nao; i++) Gj[i]=ZERO;
          }
          */
          nGi = num_Gi*2;
          bs = bidx*nthb*num_Gi;
          if (ics != ics_next) {
          for (int ii=tidx;ii<numao*3;ii+=nthb) {
            i = (ii/numao)*nao + ii%numao + minao;
            {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
            }
          } // for ii 
          } // ics != ics_next
          if (jcs != jcs_next) {
//          for (i=tidx;i<nao;i+=nthb) {
          for (i=minao+tidx;i<=maxao;i+=nthb) {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao*3 + i; // warning!
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          } // for i
          } // jcs != jcs_next
    __syncthreads();
        }
#endif /* !USE_ATOMIC */
//	klcs = klcs0;
    }		// for ( ijcs );
#ifndef USE_ATOMIC
    for (int i=tidx; i<(klcs1-klcs0)*3*3; i+=nthb) {
      int klcs, kl;
      int kcs,lcs,kao,lao;
      int kao0, lao0, k, l;
      klcs = i/9;
      kl = i - klcs*9;
      k = kl/3;
      l = kl - k*3;
      klcs += klcs0;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs] + k;
      lao = shel_ini[lcs] + l;
      G[kao*nao+lao] += Gkl[i];
    }
#endif

    return;
}


/** (pp,pp)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
NREGS_255
gpu_twoint_direct_pppp_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=1, Lb=1, Lc=1, Ld=1;
    int Lab, Lcd;
    int Labcd = 5;
    int *p_ijcounter = &ijcounter[Labcd];
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, klcs0a;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int klps0, nklps;
//    int kcs, kat, kao0, lcs, lat, lao0;
    double PPPP[3*3*3*3];
//    double PPPP[3*3*3*3], pppp;
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    int ics2, jcs2;
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
    int ijc;
    double Gij[9];
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[4];
    double *tbl = FMT_m_table4;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    double stmp[3];
    size_t sOffset = tsize;
    tsize += nthb;
#else
    double *stmp = (double *)&shared[tidx*3];
    size_t tsize = nthb*3;
    size_t sOffset = 0;
#endif
#ifdef DLB_KL_PPPP
//    __shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
//    volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif
    __shared__ double A[3];
    __shared__ double BA[3];

    mincs = MIN2(leading_cs[Lc], leading_cs[Ld]);
    minao = shel_ini[mincs];
    maxcs = MAX2(leading_cs[Lc+1], leading_cs[Ld+1]) - 1;
    maxao = shel_ini[maxcs+1]-1;
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
    Gkl = Gkl_b + bidx * num_Gkl;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao*3;
    work = work_b + bidx * nthb * WORK_SIZE;

//    nwkblk = nworkers * ndev * nblk;
//    iwkblk = workerid * ndev * nblk + idev * nblk + bidx;
//    nwkblk = nwks;
//    iwkblk = iwk + bidx;
    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;

    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lab];
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];

#ifndef USE_ATOMIC
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*6; i+=nthb ) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0)*3*3; i+=nthb) Gkl[i]=ZERO;
    __syncthreads();
#endif

#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
//    ijcsw = ijcs0 + iwk + nwks * bidx;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx;
//      ijcsw = ijcs0 + iwkblk;
      nijcsw = 0;
    }
    /*
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx * NIJCSW;
      nijcsw = NIJCSW-1;
    }
    */
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (ijcs=ijcs0+iwkblk; ijcs<ijcs1; ijcs+=nwkblk ) {
#endif
//        double A[3], BA[3];
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs1 - 1 - (ijcsw - ijcs0);
#endif
#ifdef DLB_KL_PPPP
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
        ics    = csp_ics[ijcs];
        jcs    = csp_jcs[ijcs];
        klcs0a = klcs0;
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else
        {
          float val_ab = csp_schwarz[ijcs];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
          for (int klcs = klcs0a+tidx ; klcs<=ijcs; klcs+=nthb) {
//	  for (int klcs = ijcs-tidx; klcs>=klcs0a ; klcs-=nthb) {
              int kcs, lcs;
              int iklcs;
              float dmax;
              float val_abcd = val_ab * csp_schwarz[klcs];
              if ( val_abcd < eps_ps4 ) continue;
              kcs    = csp_ics[klcs];
              lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
              if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif
#pragma unroll
        for (int i=0;i<9;i++) Gij[i] = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
        {
          /*
        double B[3];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
#pragma unroll
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        */
	BA[0] =LDG(atom_x[jat]) - (A[0]=LDG(atom_x[iat]));
	BA[1] =LDG(atom_y[jat]) - (A[1]=LDG(atom_y[iat]));
	BA[2] =LDG(atom_z[jat]) - (A[2]=LDG(atom_z[iat]));
        }
        __threadfence_block();
        __syncthreads();
	
//	for ( ; klcs<=ijcs; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs = klcs0a+tidx ; klcs<=ijcs; klcs+=nthb ) {
#else
#ifndef DLB_KL_PPPP
        for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
#else
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
        while(iklcs<nklcs) {
#endif /* DLB_KL */
#endif
            int klcs2;
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
            int ipat, ijgekl;
            double DC[3], AC[3];
#ifdef USE_INSTANT_SCHWARZ
            {
	    float val_abcd = val_ab * csp_schwarz[klcs];
            float dmax;
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
            }
#else
            int klcs;
            klcs = sklcs[iklcs];
#endif
            klcs2 = (klcs - klcs0)*9;
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao0   = LDG(shel_ini[kcs]);
	    lao0   = LDG(shel_ini[lcs]);
            {
              double C[3],D[3];
	    C[0]=LDG(atom_x[kat]); C[1]=LDG(atom_y[kat]); C[2]=LDG(atom_z[kat]);
	    D[0]=LDG(atom_x[lat]); D[1]=LDG(atom_y[lat]); D[2]=LDG(atom_z[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
            }
	    gpu_twoint_core_pppp_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      PPPP );
            {
              double Gkl2[9];
              for (int i=0;i<9;i++) Gkl2[i]=ZERO;
	    ipat = ( (ics==kcs && jcs>lcs) ? true : false );
            ijgekl = (ics>kcs);
            if (ics==kcs) ijgekl= (jcs>=lcs);
            if (!ijgekl) ipat = ( (ics==kcs && jcs<lcs) ? true : false);
#pragma unroll
	    for (int i=0, iao=iao0; i<3; i++, iao++ ) {
              int i0 = iao*nao;
              double Gik[3]={ZERO,ZERO,ZERO};
              double Gil[3]={ZERO,ZERO,ZERO};
              int I2;
		I2 = iao*(iao+1)/2;
#pragma unroll
		for (int  j=0, jao=jao0; j<3; j++, jao++ ) {
                  int j0 = jao*nao;
                  int ij = i0 + jao;
                  double coe0;
                  //double Gjk[3]={ZERO,ZERO,ZERO};
                  //double *Gjk = stmp;
                  double Gjk[3]={ZERO,ZERO,ZERO};
                  double Gjl[3]={ZERO,ZERO,ZERO};
                  int IJ;
                  //Gjk[0] = ZERO; Gjk[1] = ZERO; Gjk[2] = ZERO;
		    if ( jao>iao ) continue;
		    IJ = I2 + jao;
		    coe0 = ( iao==jao ? HALF : ONE );
#pragma unroll
		    for (int k=0, kao=kao0; k<3; k++, kao++ ) {
                      int K2;
			K2 = kao*(kao+1)/2;
#pragma unroll
			for (int l=0, lao=lao0; l<3; l++, lao++ ) {
                          double x;
			    if ( lao>kao ) continue;
			    x = PPPP[i*27+j*9+k*3+l];
			    if ( fabs(x) > eps_eri ) {
//				int ij, ik, il, jk, jl, kl, i0, j0;
				int kl;
                                int KL;
				KL = K2 + lao;
//				if ( IJ >= KL  || ipat ) {
				if ((ijgekl&&IJ >= KL) || (!ijgekl&&KL>=IJ) || ipat ) {
                                  double x4, dx;
                                  /*
				    coe = coe0;
				    if ( kao==lao ) coe *= HALF;
				    if ( KL == IJ ) coe *= HALF;
				    x  = coe * pppp;
                                  */
				    x *= coe0;
				    if ( kao==lao ) x *= HALF;
				    if ( KL == IJ ) x *= HALF;
                                    x4 = 4.0 * x;
                                    x *= -x_coef;

                                    /*
				    x4 = 4.e0 * x;
				    i0 = iao*nao;
				    j0 = jao*nao;
				    ij = i0 + jao;
				    ik = i0 + kao;
				    il = i0 + lao;
				    jk = j0 + kao;
				    jl = j0 + lao;
                                    */
				    kl = kao*nao + lao;
                                    /*
				    G[ij] += x4*Ds[kl];
				    G[kl] += x4*Ds[ij];
				    G[ik] -=  x*Ds[jl];
				    G[il] -=  x*Ds[jk];
				    G[jk] -=  x*Ds[il];
				    G[jl] -=  x*Ds[ik];
                                    */
                                    Gik[k] += x*LDG(Ds[j0+lao]);
                                    Gil[l] += x*LDG(Ds[j0+kao]);
                                    Gjk[k] += x*LDG(Ds[i0+lao]);
                                    Gjl[l] += x*LDG(Ds[i0+kao]);

                                    Gij[i*3+j]  += x4*LDG(Ds[kl]);
                                    Gkl2[k*3+l] += x4*LDG(Ds[ij]);
				}
			    } // if ( fabs )
			} // for ( lao )
		    } // for ( kao )
#ifdef USE_ATOMIC
#pragma unroll
                  for (int k=0; k<3; k++) {
                    int jk = jao*nao + kao0 + k;
                    int jl = jao*nao + lao0 + k;
                    if (kcs==lcs) Gjk[k] += Gjl[k];
                    else atomicAdd(&G[jl], Gjl[k]);
                    atomicAdd(&G[jk], Gjk[k]);
                  }
#else
#pragma unroll
                  for (int k=0; k<3; k++) {
                    Gj[j*nao+lao0+k] += Gjl[k];
                    Gj[j*nao+kao0+k] += Gjk[k];
                  }
#endif
		} // for ( jao )
#ifdef USE_ATOMIC
#pragma unroll
              for (int k=0; k<3; k++) {
                int ik = iao*nao + kao0 + k;
                int il = iao*nao + lao0 + k;
                if (kcs==lcs) Gik[k] += Gil[k];
                else atomicAdd(&G[il], Gil[k]);
                atomicAdd(&G[ik], Gik[k]);
              }
#else
#pragma unroll
              for (int k=0; k<3; k++) {
                Gi[i*nao+kao0+k] += Gik[k];
                Gi[i*nao+lao0+k] += Gil[k];
              }
#endif
	    }	// for ( iao )
#ifdef USE_ATOMIC
#pragma unroll
            for (int k=0, kao=kao0; k<3; k++, kao++ ) 
#pragma unroll
              for (int l=0, lao=lao0; l<3; l++, lao++ ) 
                atomicAdd(&G[kao*nao+lao],Gkl2[k*3+l]);
#else
#pragma unroll
            for (int kl=0; kl<9; kl++) 
              Gkl[klcs2+kl] += Gkl2[kl];
#endif
          } // block for additions
#ifdef DLB_KL_PPPP
          if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
          iklcs = klcsw[widx] + tidw;
          }; // while (iklcs)
#endif
	}	// for ( klcs );
#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
//            ijc++;
//            ijcsw = ijcs0 + iwk + ijc*nwks;
//            ijcsw = ijcs0 + ijc*nwks + (iwk + ijc*3)%nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
//            ijc = atomicAdd(p_ijcounter,NIJCSW);
//            ijcsw = ijcs0 + iwk + ijc*nwks;
//            ijcsw = ijcs0 + ijc*nwks + (iwk + ijc*3)%nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
        {
          double *sGij0 = (double *)&shared[sOffset+tidx];
          __syncthreads();
#pragma unroll
          for (int i=0,iao=iao0;i<3;i++,iao++) {
#pragma unroll
            for (int j=0,jao=jao0; j<3; j++,jao++) {
              //atomicAdd(&G[iao*nao+jao],Gij[i*3+j]);
              {
                double *sGij = &shared[sOffset+widx*WARP_SIZE];
                *sGij0 = Gij[i*3+j];
                warpReduce(sGij, tidw);
              }
              __syncthreads();
              if (tidx==0) { // Reduce Gij for warp masters
                double *sGij = &shared[sOffset];
                double dtmp = sGij[0];
                for (int k=1; k<nwrp; k++) dtmp += sGij[k*WARP_SIZE];
                //G[iao*nao+jao] += dtmp;
                atomicAdd(&G[iao*nao+jao], dtmp);
              }
              __syncthreads();
            }
          }
        }
        __syncthreads();
#else
//	klcs = klcs0;
//        if (n2ei>0) {
        {
          int j;
          int nGi;
          int bs;
	  int i0 = iao0*nao;
	  int j0 = jao0*nao;
          int ijcs_next;
          int ics_next=-1, jcs_next=-1;
#pragma unroll
          for (int i=0;i<3;i++) {
#pragma unroll
            for (int j=0,jao=jao0; j<3; j++,jao++) {
              Gi[i*nao+jao] += Gij[i*3+j];
            }
          }
    __syncthreads();
#ifdef GPU_DLB
//          ijcs_next = ijcsw;
          ijcs_next = ijcs1 - ijcsw + 1;
          ijcs_next = ijcs1 - 1 - (ijcsw - ijcs0);
          if (ijcsw<ijcs1) {
#else
          ijcs_next = ijcs+nwkblk;
          if (ijcs_next<ijcs1) {
#endif
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
          nGi = num_Gi*2;
          bs = bidx*nthb*num_Gi;
          if (ics != ics_next) {
//          for (int ii=tidx;ii<nao*3;ii+=nthb) {
//            i = (ii/nao)*nao + ii%nao;
          for (int ii=tidx;ii<numao*3;ii+=nthb) {
            int i = (ii/numao)*nao + ii%numao + minao;
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
          } 
          }
         // if (jcs != jcs_next) {
          if (jcs != jcs_next) {
//          for (i=tidx;i<nao;i+=nthb) {
//          for (i=minao+tidx;i<=maxao;i+=nthb) {
          for (int ii=tidx;ii<numao*3;ii+=nthb) {
            int i = (ii/numao)*nao + ii%numao + minao;
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao*3 + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          } 
          }
    __syncthreads();
//	  for ( i=0; i<nao*3; i++ ) Gi[i]=ZERO;
//	  for ( i=0; i<nao; i++ ) Gj[i]=ZERO;
        }
#endif /* USE_ATOMIC */
//	klcs = klcs0;
    }		// for ( ijcs );

#ifndef USE_ATOMIC
    for (int i=tidx; i<(klcs1-klcs0)*3*3; i+=nthb) {
      int klcs, kl;
      int kcs,lcs,kao,lao;
      int kao0, lao0, k, l;
      klcs = i/9;
      kl = i - klcs*9;
      k = kl/3;
      l = kl - k*3;
      klcs += klcs0;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs] + k;
      lao = shel_ini[lcs] + l;
      G[kao*nao+lao] += Gkl[i];
    }
#endif

    return;
}

/** (ds,ss)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
NREGS_128
gpu_twoint_direct_dsss_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=2, Lb=0, Lc=0, Ld=0;
    int Lab, Lcd;
    int Labcd = 6;
    int *p_ijcounter = &ijcounter[Labcd];
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, klcs0a;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, lcs, lat, lao;
    double DSSS[6];
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
//    double coe;
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    int ics2, jcs2;
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int minao1[2], maxao1[2];
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[2];
    double *tbl = FMT_m_table2;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    double Gij[1];
    //double Gil[1];
    //double Gjl[1];
    //double *Gij = &shared[tsize+tidx];
    //double *Gij = &shared[tsize+1+tidx+nthb];
    //double *Gij = &shared[tsize+1+tidx+nthb*2];
    size_t sOffset = tsize;
    tsize += nthb;
#else
    size_t sOffset = 0;
    double *Gij = &shared[tidx];
    //double *Gil = &shared[tidx+nthb];
    //double *Gjl = &shared[tidx+nthb*2];
    size_t tsize = nthb*3;
#endif
#ifdef DLB_KL_DSSS
//    __shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
//    volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif
    __shared__ double A[3];
    __shared__ double BA[3];

    minao1[0] = shel_ini[leading_cs[La]];
    minao1[1] = shel_ini[leading_cs[Lb]];
    minao = MIN2(minao1[0], minao1[1]);
    maxao1[0] = shel_ini[leading_cs[La+1]] - 1;
    maxao1[1] = shel_ini[leading_cs[Lb+1]] - 1;
    maxao = MAX2(maxao1[0], maxao1[1]);
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
    Gkl = Gkl_b + bidx * num_Gkl;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao;
    work = work_b + bidx * nthb * WORK_SIZE;

//    nwkblk = nworkers * ndev * nblk;
//    iwkblk = workerid * ndev * nblk + idev * nblk + bidx;
//    nwkblk = nwks;
//    iwkblk = iwk + bidx;
    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;

    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lcd];
    ijcs1 = leading_cs_pair[Lcd+1];
    klcs0 = leading_cs_pair[Lab];
    klcs1 = leading_cs_pair[Lab+1];

#ifndef USE_ATOMIC
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*2; i+=nthb ) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0)*6; i+=nthb) Gkl[i]=ZERO;
    __syncthreads();
#endif

#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
//    ijcsw = ijcs0 + iwk + nwks * bidx;
//    nijcsw = 0;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx * NIJCSW;
      nijcsw = NIJCSW-1;
    }
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (int ijcs2=ijcs0+iwkblk; ijcs2<ijcs1; ijcs2+=nwkblk ) {
#endif
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs0 + ijcs1 - ijcsw - 1;
        /*
        __syncthreads();
        if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
        */
#else
      ijcs = ijcs2;
#endif
#ifdef SORT_IJ_SCHWARZ
      ijcs = sorted_csp[ijcs];
#endif
#ifdef DLB_KL_DSSS
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
//        if(tidx==0) printf("%d:0 %d %d %d\n",tidx,ijcs,ics,jcs);
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else /* !USE_INSTANT_SCHWARZ */
        {
	  float val_ab = csp_schwarz[ijcs];
//          int sklcs_l[NKLBUF];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
	  for (int klcs2 = klcs0+tidx ; klcs2<klcs1; klcs2+=nthb) {
//	  for (int klcs2 = klcs1-tidx-1; klcs2>=klcs0 ; klcs2-=nthb) {
              int klcs=klcs2;
              int kcs, lcs;
              int iklcs;
              float dmax;
	      float val_abcd;
#ifdef SORT_KL_SCHWARZ
          klcs = sorted_csp[klcs2];
#endif
	      val_abcd = val_ab * csp_schwarz[klcs];
	      if ( val_abcd < eps_ps4 ) continue;
	      kcs    = csp_ics[klcs];
	      lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	      if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
#ifdef SORT_KL_SCHWARZ
              klcs = klcs2;
#endif
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
//          if(bidx==0&&tidx==0) printf("nklcs = %d\n", nklcs);
        }
#endif /* USE_INSTANT_SCHWARZ */
        *Gij  = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao    = shel_ini[ics];
	jao    = shel_ini[jcs];
        __syncthreads();
        if(tidx == 0) {
        /*
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        */
        BA[0] = LDG(atom_x[jat]) + (A[0] = -LDG(atom_x[iat]));
        BA[1] = LDG(atom_y[jat]) + (A[1] = -LDG(atom_y[iat]));
        BA[2] = LDG(atom_z[jat]) + (A[2] = -LDG(atom_z[iat]));
        }
        __threadfence_block();
        __syncthreads();

//        if(tidx==0) printf("%d:k %d %d\n",tidx,klcs0,klcs1);
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
//	for (klcs = klcs0 ; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (klcs = klcs0+tidx ; klcs<klcs1; klcs+=nthb ) {
#ifdef DLB_KL_DSSS
  NOT supported combination of defines.
#endif
#else /* !USE_INSTANT_SCHWARZ */
#ifndef DLB_KL_DSSS
//	for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
	for (int iklcs2 = tidx ; iklcs2<nklcs; iklcs2+=nthb ) {
          int iklcs;
          int it = iklcs2 / nthb;
          if (it%2 == 0) iklcs = iklcs2;
          else {
            int ilast = MIN2(nklcs, (it+1)*nthb);
            iklcs = it*nthb + ilast - iklcs2 - 1;
          }
#else /* !DLB_KL_DSSS */
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
        while(iklcs<nklcs) {
#endif /* !DLB_KL_DSSS */
#endif /* !USE_INSTANT_SCHWARZ */
            double DC[3], AC[3];
            float dmax;
            int jk, jl, j0;
            int kao0, lao0, kao, lao;
//            int n2ei2 = 0;
            int klcs2;
#ifdef USE_INSTANT_SCHWARZ
	    float val_abcd = val_ab * csp_schwarz[klcs];
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
#else
            klcs = sklcs[iklcs];
#ifdef SORT_KL_SCHWARZ
            klcs = sorted_csp[klcs];
#endif
#endif /* USE_INSTANT_SCHWARZ */
            klcs2 = klcs - klcs0;
//         if (tidx==0) printf("%d:kl %d %d\n",tidx,klcs, klcs2);
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
//            Gjk = Gjl = ZERO;
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao0   = LDG(shel_ini[kcs]);
	    lao    = LDG(shel_ini[lcs]);
            /*
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
            */
            {
              double dtmp;
              AC[0] = A[0] + (dtmp = LDG(atom_x[kat]));
              DC[0] = LDG(atom_x[lat]) - dtmp;
              AC[1] = A[1] + (dtmp = LDG(atom_y[kat]));
              DC[1] = LDG(atom_y[lat]) - dtmp;
              AC[2] = A[2] + (dtmp = LDG(atom_z[kat]));
              DC[2] = LDG(atom_z[lat]) - dtmp;
            }

/*
	    gpu_twoint_core_dsss_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DSSS );
*/
// (ss,ds) -> (ds,ss)
	    //gpu_twoint_core_dsss_(
	    gpu_twoint_core_os_dsss(
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,   AC,      DSSS );

            {
	    double coe = ( iao == jao ? HALF : ONE );
            //*Gil  = ZERO;
            //*Gjl  = ZERO;
            double Gil  = ZERO;
            double Gjl  = ZERO;
#pragma unroll
	    for (int k=0, kao=kao0; k<6; k++, kao++ ) {
		if ( fabs(DSSS[k]) > eps_eri ) {
		    double x, x4;
                    double dx;
		    int ij, ik, il, jk, jl, kl, i0, j0;
		    x  = coe * DSSS[k];

		    x4 = 4.e0 * x;
                    x *= -x_coef;
		    i0 = iao*nao;
		    j0 = jao*nao;
		    ij = i0 + jao;
		    ik = i0 + kao;
		    il = i0 + lao;
		    jk = j0 + kao;
		    jl = j0 + lao;
		    kl = kao*nao + lao;
                    /*
		    G[ij] += x4*Ds[kl];
		    G[kl] += x4*Ds[ij];
		    G[ik] -=  x*Ds[jl];
		    G[il] -=  x*Ds[jk];
		    G[jk] -=  x*Ds[il];
		    G[jl] -=  x*Ds[ik];
                    */
		    *Gij += x4*LDG(Ds[kl]);
		    Gil += x*LDG(Ds[jk]);
		    Gjl += x*LDG(Ds[ik]);
#ifdef USE_ATOMIC
		    dx = x4*LDG(Ds[ij]);
		    atomicAdd(&G[kl], dx);
		    dx = x*LDG(Ds[jl]);
		    atomicAdd(&G[ik], dx);
		    dx = x*LDG(Ds[il]);
		    atomicAdd(&G[jk], dx);
#else
                    Gkl[klcs2*6+k] += x4*LDG(Ds[ij]);
                    Gi[kao] += x*LDG(Ds[jl]);
                    Gj[kao] += x*LDG(Ds[il]);
#endif
		}
	    }	// for ( k, kao);
#ifdef USE_ATOMIC
            atomicAdd(&G[iao*nao+lao], Gil);
            atomicAdd(&G[jao*nao+lao], Gjl);
#else
            Gi[lao] += Gil;
            Gj[lao] += Gjl;
#endif
            } // block for additions
#ifdef DLB_KL_DSSS
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx] + tidw;
          } // while (iklcs)
#endif
	}	// for ( klcs );
//	klcs = klcs0;
#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
#if 0
        atomicAdd(&G[iao*nao+jao], *Gij);
#else
        shared[sOffset+tidx] = *Gij;
          { // Reduce Gij within warp
            //double *sGij = &shared[widx*WARP_SIZE];
            double *sGij = &shared[sOffset+widx*WARP_SIZE];
            warpReduce(sGij, tidw);
          }
          __syncthreads();
          if (tidx==0) { // Reduce Gij for warp masters
            //double *sGij = &shared[0];
            double *sGij = &shared[sOffset];
            double dtmp = sGij[0];
            for (int j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
            atomicAdd(&G[iao*nao+jao], dtmp);
          }
#endif
        __syncthreads();
#else /* !USE_ATOMIC */
        Gi[jao] += *Gij;
        {
          int i, j;
          int nGi;
          int bs;
	  int i0 = iao*nao;
	  int j0 = jao*nao;
          int ijcs_next;
          int ics_next, jcs_next;
          ics_next=jcs_next=-1;
#ifdef GPU_DLB
          __syncthreads();
//          ijcs_next = ijcsw;
//          if (ijcs_next<ijcs1) { 
          ijcs_next = ijcs0 + ijcs1 - ijcsw - 1;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
//          if (ijcs_next>=ijcs0) { 
          if (ijcsw<ijcs1) { 
#else /* !GPU_DLB */
          ijcs_next = ijcs2+nwkblk;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
          if (ijcs_next<ijcs1) { 
#endif /* !GPU_DLB */
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
#ifndef GPU_DLB
          if (ics != ics_next || jcs != jcs_next) __syncthreads();
#endif
          nGi = num_Gi*2;
          bs = bidx*nthb*num_Gi;
          if (ics != ics_next) {
#ifdef ADD_FULL_NAO
          for (i=minao+tidx;i<=maxao;i+=nthb) {
#else
          for (int k=0;k<2;k++) {
          for (i=minao1[k]+tidx;i<=maxao1[k];i+=nthb) {
#endif
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
          } // for i
#ifndef ADD_FULL_NAO
          } // for k
#endif
          } // ics != ics_next
          if (jcs != jcs_next) {
//          for (i=tidx;i<nao;i+=nthb) {
#ifdef ADD_FULL_NAO
          for (i=minao+tidx;i<=maxao;i+=nthb) {
#else
          for (int k=0;k<2;k++) {
          for (i=minao1[k]+tidx;i<=maxao1[k];i+=nthb) {
#endif
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          } // for i
#ifndef ADD_FULL_NAO
          } // for k
#endif
          } // jcs != jcs_next
    __syncthreads();
        }
#endif /* !USE_ATOMIC */
    }		 // for ( ijcs );

#ifndef USE_ATOMIC
    for (int i=tidx; i<(klcs1-klcs0)*6; i+=nthb) {
      int klcs, k;
      int kcs,lcs,kao,lao;
      klcs = i/6;
      k = i - klcs*6;
      klcs += klcs0;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs] + k;
      lao = shel_ini[lcs];
      G[kao*nao+lao] += Gkl[i];
    }
#endif

    return;
}

/** (ds,ps)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
NREGS_128
gpu_twoint_direct_dsps_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=2, Lb=0, Lc=1, Ld=0;
    int Lab, Lcd;
    int Labcd = 7;
    int *p_ijcounter = &ijcounter[Labcd];
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, klcs0a;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao0, jcs, jat, jao0;
    int kcs, kat, kao0, lcs, lat, lao0;
    double DSPS[6*3];
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    int ics2, jcs2;
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int minao1[2], maxao1[2];
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[3];
    double *tbl = FMT_m_table3;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    size_t sOffset = tsize;
    double Gij[3];
    //double *Gij = &shared[sOffset+tidx*3];
    tsize += nthb;
#else
    size_t sOffset = 0;
    double *Gij = &shared[tidx*3];
    size_t tsize = nthb*3;
#endif
#ifdef DLB_KL_DSPS
//    __shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
//    volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif
    __shared__ double A[3];
    __shared__ double BA[3];

    minao1[0] = shel_ini[leading_cs[La]];
    minao1[1] = shel_ini[leading_cs[Lb]];
    minao = MIN2(minao1[0], minao1[1]);
    maxao1[0] = shel_ini[leading_cs[La+1]] - 1;
    maxao1[1] = shel_ini[leading_cs[Lb+1]] - 1;
    maxao = MAX2(maxao1[0], maxao1[1]);
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
    Gkl = Gkl_b + bidx * num_Gkl;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao*3;
    work = work_b + bidx * nthb * WORK_SIZE;

//    nwkblk = nworkers * ndev * nblk;
//    iwkblk = workerid * ndev * nblk + idev * nblk + bidx;
//    nwkblk = nwks;
//    iwkblk = iwk + bidx;
    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;

    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lcd];
    ijcs1 = leading_cs_pair[Lcd+1];
    klcs0 = leading_cs_pair[Lab];
    klcs1 = leading_cs_pair[Lab+1];

#ifndef USE_ATOMIC
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*4; i+=nthb ) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0)*6*1; i+=nthb) Gkl[i]=ZERO;
    __syncthreads();
#endif

#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
//    ijcsw = ijcs0 + iwk + nwks * bidx;
//    nijcsw = 0;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx * NIJCSW;
      nijcsw = NIJCSW-1;
    }
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (int ijcs2=ijcs0+iwkblk; ijcs2<ijcs1; ijcs2+=nwkblk ) {
#endif
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs0 + ijcs1 - ijcsw - 1;
        /*
        __syncthreads();
        if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
        */
#else
      ijcs = ijcs2;
#endif
#ifdef SORT_IJ_SCHWARZ
      ijcs = sorted_csp[ijcs];
#endif
#ifdef DLB_KL_DSPS
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
//        if(tidx==0) printf("%d:0 %d %d %d\n",tidx,ijcs,ics,jcs);
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else /* !USE_INSTANT_SCHWARZ */
        {
	  float val_ab = csp_schwarz[ijcs];
//          int sklcs_l[NKLBUF];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
	  for (int klcs2 = klcs0+tidx ; klcs2<klcs1; klcs2+=nthb) {
//	  for (int klcs2 = klcs1-tidx-1; klcs2>=klcs0 ; klcs2-=nthb) {
              int klcs=klcs2;
              int kcs, lcs;
              int iklcs;
              float dmax;
	      float val_abcd;
#ifdef SORT_KL_SCHWARZ
          klcs = sorted_csp[klcs2];
#endif
	      val_abcd = val_ab * csp_schwarz[klcs];
	      if ( val_abcd < eps_ps4 ) continue;
	      kcs    = csp_ics[klcs];
	      lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	      if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
#ifdef SORT_KL_SCHWARZ
              klcs = klcs2;
#endif
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
//          if(bidx==0&&tidx==0) printf("nklcs = %d\n", nklcs);
        }
#endif /* USE_INSTANT_SCHWARZ */
#pragma unroll
        for (int i=0;i<3;i++) Gij[i] = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
        __syncthreads();
        if(tidx == 0) {
        /*
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        */
        BA[0] = LDG(atom_x[jat]) + (A[0] = -LDG(atom_x[iat]));
        BA[1] = LDG(atom_y[jat]) + (A[1] = -LDG(atom_y[iat]));
        BA[2] = LDG(atom_z[jat]) + (A[2] = -LDG(atom_z[iat]));
        }
        __threadfence_block();
        __syncthreads();

//        if(tidx==0) printf("%d:k %d %d\n",tidx,klcs0,klcs1);
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
	
//	for (klcs = klcs0 ; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (klcs = klcs0+tidx ; klcs<klcs1; klcs+=nthb ) {
#ifdef DLB_KL_DSPS
  NOT supported combination of defines.
#endif
#else /* !USE_INSTANT_SCHWARZ */
#ifndef DLB_KL_DSPS
//	for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
	for (int iklcs2 = tidx ; iklcs2<nklcs; iklcs2+=nthb ) {
          int iklcs;
          int it = iklcs2 / nthb;
          if (it%2 == 0) iklcs = iklcs2;
          else {
            int ilast = MIN2(nklcs, (it+1)*nthb);
            iklcs = it*nthb + ilast - iklcs2 - 1;
          }
#else /* !DLB_KL_DSPS */
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
        while(iklcs<nklcs) {
#endif /* !DLB_KL_DSPS */
#endif /* !USE_INSTANT_SCHWARZ */
            double DC[3], AC[3];
            float dmax;
//            int n2ei2 = 0;
            int klcs2;
#ifdef USE_INSTANT_SCHWARZ
	    float val_abcd = val_ab * csp_schwarz[klcs];
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
#else
            klcs = sklcs[iklcs];
#ifdef SORT_KL_SCHWARZ
            klcs = sorted_csp[klcs];
#endif
#endif /* USE_INSTANT_SCHWARZ */
            klcs2 = klcs - klcs0;
//         if (tidx==0) printf("%d:kl %d %d\n",tidx,klcs, klcs2);
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao0   = LDG(shel_ini[kcs]);
	    lao0   = LDG(shel_ini[lcs]);
            /*
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
            */
            {
              double dtmp;
              AC[0] = A[0] + (dtmp = LDG(atom_x[kat]));
              DC[0] = LDG(atom_x[lat]) - dtmp;
              AC[1] = A[1] + (dtmp = LDG(atom_y[kat]));
              DC[1] = LDG(atom_y[lat]) - dtmp;
              AC[2] = A[2] + (dtmp = LDG(atom_z[kat]));
              DC[2] = LDG(atom_z[lat]) - dtmp;
            }
/*
	    gpu_twoint_core_dsps_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DSPS );
*/
// (ps,ds) -> (ds,ps)
	    //gpu_twoint_core_dsps_(
#ifdef GPU_TWOINT_RYS
	    gpu_twoint_core_rys_dsps
#else
	    gpu_twoint_core_os_dsps
#endif
                   (&nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,   AC,      DSPS);
            {
              int iao, jao, kao, lao;
              int ix = 0;
              double Gjl = ZERO;
              double Gil[3] = {ZERO,ZERO,ZERO};
              jao = jao0;
              lao = lao0;
#pragma unroll
              for (int k=0, kao=kao0; k<6; k++, kao++ ) {
                double Gjk = ZERO;
                double Gkl2 = ZERO;
#pragma unroll
		for (int i=0, iao=iao0; i<3; i++, iao++, ix++ ) {
		    if ( fabs(DSPS[ix]) > eps_eri ) {
			double x, x4;
                        double dx;
			int ij, ik, il, jk, jl, kl, i0, j0;
			x  = DSPS[ix];

			x4 = 4.e0 * x;
                        x *= -x_coef;
			i0 = iao*nao;
			j0 = jao*nao;
			ij = i0 + jao;
			ik = i0 + kao;
			il = i0 + lao;
			jk = j0 + kao;
			jl = j0 + lao;
			kl = kao*nao + lao;
                        /*
			G[ij] += x4*Ds[kl];
			G[kl] += x4*Ds[ij];
			G[ik] -=  x*Ds[jl];
			G[il] -=  x*Ds[jk];
			G[jk] -=  x*Ds[il];
			G[jl] -=  x*Ds[ik];
                        */
                        Gij[i] += x4*LDG(Ds[kao*nao+lao]);
			Gjk += x*LDG(Ds[iao*nao+lao]);
			Gjl += x*LDG(Ds[iao*nao+kao]);
			Gil[i] += x*LDG(Ds[jao*nao+kao]);
                        Gkl2 += x4*LDG(Ds[iao*nao+jao]);
#ifdef USE_ATOMIC
			dx = x*LDG(Ds[jl]);
                        atomicAdd(&G[ik], dx);
#else
                        Gi[i*nao+kao] += x*LDG(Ds[jao*nao+lao]);
#endif
		    }
		}  // for (i, iao);
#ifdef USE_ATOMIC
                atomicAdd(&G[jao*nao+kao], Gjk);
                atomicAdd(&G[kao*nao+lao], Gkl2);
#else
                Gj[kao] += Gjk;
                Gkl[klcs2*6+k] += Gkl2;
#endif
	    }	// for ( k, kao);
#ifdef USE_ATOMIC
            atomicAdd(&G[jao*nao+lao], Gjl);
#pragma unroll
            for (int i=0,iao=iao0;i<3;i++,iao++)
              atomicAdd(&G[iao*nao+lao], Gil[i]);
#else
            Gj[lao] += Gjl;
#pragma unroll
            for (int i=0,iao=iao0;i<3;i++,iao++)
              Gi[i*nao+lao] += Gil[i];
#endif
          } // block for additions
#ifdef DLB_KL_DSPS
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx] + tidw;
          } // while (iklcs)
#endif
	}	// for ( klcs );
//	klcs = klcs0;

#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
#if 0
#pragma unroll
        for (int i=0; i<3; i++)
          atomicAdd(&G[jao0*nao+iao0+i], Gij[i]);
#else
        {
#ifndef CUDA_FMT_M_SM
          double tGij[3];
          for (int i=0; i<3; i++) tGij[i] = Gij[i];
#else
          double *tGij = Gij;
#endif
          for (int i=0; i<3; i++) {
            double *ss = &shared[sOffset+tidx];
            __syncthreads();
            *ss = tGij[i];
            { // Reduce Gij within warp
              double *sGij = &shared[sOffset+widx*WARP_SIZE];
              warpReduce(sGij, tidw);
            }
            __syncthreads();
            if (tidx==0) { // Reduce Gij for warp masters
              double *sGij = &shared[sOffset];
              double dtmp = sGij[0];
              for (int j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
              atomicAdd(&G[jao0*nao+iao0+i], dtmp);
            }
          } // i
        }
#endif
        __syncthreads();
#else
#pragma unroll
        for (int i=0; i<3; i++) Gi[i*nao+jao0] += Gij[i];
        {
          int i, j;
          int nGi;
          int bs;
	  int i0 = iao0*nao;
	  int j0 = jao0*nao;
          int ijcs_next;
          int ics_next, jcs_next;
          ics_next=jcs_next=-1;
#ifdef GPU_DLB
          __syncthreads();
//          ijcs_next = ijcsw;
//          if (ijcs_next<ijcs1) { 
          ijcs_next = ijcs0 + ijcs1 - ijcsw - 1;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
//          if (ijcs_next>=ijcs0) { 
          if (ijcsw<ijcs1) { 
#else /* !GPU_DLB */
          ijcs_next = ijcs2+nwkblk;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
          if (ijcs_next<ijcs1) { 
#endif /* !GPU_DLB */
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
#ifndef GPU_DLB
          if (ics != ics_next || jcs != jcs_next) __syncthreads();
#endif
          nGi = num_Gi*2;
          bs = bidx*nthb*num_Gi;
          if (ics != ics_next) {
#ifdef ADD_FULL_NAO
          //for (i=minao+tidx;i<=maxao;i+=nthb) {
          for (int ii=tidx;ii<numao*3;ii+=nthb) {
            i = (ii/numao)*nao + ii%numao + minao;
#else
          for (int k=0;k<2;k++) {
//          for (i=minao1[k]+tidx;i<=maxao1[k];i+=nthb) {
          int numao1=maxao1[k]-minao1[k]+1;
          for (int ii=tidx;ii<numao1*3;ii+=nthb) {
            i = (ii/numao1)*nao + ii%numao1 + minao1[k];
#endif
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
          } // for i
#ifndef ADD_FULL_NAO
          } // for k
#endif
          } // ics != ics_next
          if (jcs != jcs_next) {
//          for (i=tidx;i<nao;i+=nthb) {
#ifdef ADD_FULL_NAO
          for (i=minao+tidx;i<=maxao;i+=nthb) {
#else
          for (int k=0;k<2;k++) {
          for (i=minao1[k]+tidx;i<=maxao1[k];i+=nthb) {
#endif
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao*3 + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          } // for i
#ifndef ADD_FULL_NAO
          } // for k
#endif
          } // jcs != jcs_next
    __syncthreads();
        }
#endif /* !USE_ATOMIC */
    }		// for ( ijcs );

#ifndef USE_ATOMIC
    for (int i=tidx; i<(klcs1-klcs0)*6; i+=nthb) {
      int klcs, k;
      int kcs,lcs,kao,lao;
      klcs = i/6;
      k = i - klcs*6;
      klcs += klcs0;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs] + k;
      lao = shel_ini[lcs];
      G[kao*nao+lao] += Gkl[i];
    }
#endif

    return;
}

/** (ds,pp)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
NREGS_255
gpu_twoint_direct_dspp_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=2, Lb=0, Lc=1, Ld=1;
    int Lab, Lcd;
    int Labcd = 8;
    int *p_ijcounter = &ijcounter[Labcd];
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, klcs0a;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao0, jcs, jat, jao0;
    int kcs, kat, kao0, lcs, lat, lao0;
    double DSPP[6*3*3];
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    int ics2, jcs2;
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int minao1[2], maxao1[2];
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
//    double *Gij = &shared[tidx*3];
    double Gij[6*1];
//    double *stmp = (double *)&shared[tidx*3];
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[4];
    double *tbl = FMT_m_table4;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    size_t sOffset = tsize;
    tsize += nthb;
#else
    size_t tsize = nthb*3;
    size_t sOffset = 0;
#endif
#ifdef DLB_KL_DSPP
//    __shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
//    volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif
    __shared__ double A[3];
    __shared__ double BA[3];

    mincs = MIN2(leading_cs[Lc], leading_cs[Ld]);
    minao = shel_ini[mincs];
    maxcs = MAX2(leading_cs[Lc+1], leading_cs[Ld+1]) - 1;
    maxao = shel_ini[maxcs+1]-1;
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
    Gkl = Gkl_b + bidx * num_Gkl;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao*6;
    work = work_b + bidx * nthb * WORK_SIZE;

//    nwkblk = nworkers * ndev * nblk;
//    iwkblk = workerid * ndev * nblk + idev * nblk + bidx;
//    nwkblk = nwks;
//    iwkblk = iwk + bidx;
    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;

    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lab];
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];

#ifndef USE_ATOMIC
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*(6+1); i+=nthb ) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0)*3*3; i+=nthb) Gkl[i]=ZERO;
    __syncthreads();
#endif

#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
//    ijcsw = ijcs0 + iwk + nwks * bidx;
//    nijcsw = 0;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx * NIJCSW;
      nijcsw = NIJCSW-1;
    }
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (int ijcs2=ijcs0+iwkblk; ijcs2<ijcs1; ijcs2+=nwkblk ) {
#endif
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs0 + ijcs1 - ijcsw - 1;
        /*
        __syncthreads();
        if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
        */
#else
      ijcs = ijcs2;
#endif
#ifdef SORT_IJ_SCHWARZ
      ijcs = sorted_csp[ijcs];
#endif
#ifdef DLB_KL_DSPP
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
        klcs0a = klcs0;
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else
        {
	  float val_ab = csp_schwarz[ijcs];
//          int sklcs_l[NKLBUF];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
	  for (int klcs2 = klcs0a+tidx ; klcs2<klcs1; klcs2+=nthb) {
//	  for (int klcs2 = klcs1-tidx-1; klcs2>=klcs0a ; klcs2-=nthb) {
              int klcs=klcs2;
              int kcs, lcs;
              int iklcs;
              float dmax;
	      float val_abcd;
#ifdef SORT_KL_SCHWARZ
          klcs = sorted_csp[klcs2];
#endif
	      val_abcd = val_ab * csp_schwarz[klcs];
	      if ( val_abcd < eps_ps4 ) continue;
	      kcs    = csp_ics[klcs];
	      lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	      if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
#ifdef SORT_KL_SCHWARZ
              klcs = klcs2;
#endif
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
//          if(bidx==0&&tidx==0) printf("nklcs = %d\n", nklcs);
        }
#endif /* USE_INSTANT_SCHWARZ */
#pragma unroll
        for (int i=0;i<1*6;i++) Gij[i] = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
        /*
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        */
        BA[0] = LDG(atom_x[jat]) - (A[0] = LDG(atom_x[iat]));
        BA[1] = LDG(atom_y[jat]) - (A[1] = LDG(atom_y[iat]));
        BA[2] = LDG(atom_z[jat]) - (A[2] = LDG(atom_z[iat]));

        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        klcs0a = klcs0;
//	for (klcs = klcs0a ; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (klcs = klcs0a+tidx ; klcs<klcs1; klcs+=nthb ) {
#ifdef DLB_KL_DSPP
  NOT supported combination of defines.
#endif
#else
#ifndef DLB_KL_DSPP
//	for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
	for (int iklcs2 = tidx ; iklcs2<nklcs; iklcs2+=nthb ) {
          int iklcs;
          int it = iklcs2 / nthb;
          if (it%2 == 0) iklcs = iklcs2;
          else {
            int ilast = MIN2(nklcs, (it+1)*nthb);
            iklcs = it*nthb + ilast - iklcs2 - 1;
          }
#else
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
        while(iklcs<nklcs) {
#endif /* DLB_KL */
#endif /* USE_INSTANT_SCHWARZ */
            double DC[3], AC[3];
            float dmax;
            int jk, jl, j0;
//            int n2ei2 = 0;
            int klcs2;
#ifdef USE_INSTANT_SCHWARZ
	    float val_abcd = val_ab * csp_schwarz[klcs];
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
#else
            klcs = sklcs[iklcs];
#ifdef SORT_KL_SCHWARZ
            klcs = sorted_csp[klcs];
#endif
#endif /* USE_INSTANT_SCHWARZ */
            klcs2 = klcs - klcs0;
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao0   = LDG(shel_ini[kcs]);
	    lao0   = LDG(shel_ini[lcs]);
            /*
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
            */
            {
            double dtmp;
            AC[0] = A[0] - (dtmp = LDG(atom_x[kat]));
            DC[0] = LDG(atom_x[lat]) - dtmp;
            AC[1] = A[1] - (dtmp = LDG(atom_y[kat]));
            DC[1] = LDG(atom_y[lat]) - dtmp;
            AC[2] = A[2] - (dtmp = LDG(atom_z[kat]));
            DC[2] = LDG(atom_z[lat]) - dtmp;
            }
	    //gpu_twoint_core_dspp_(
	    gpu_twoint_core_os_dspp(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DSPP );
            {
              int jao=jao0;
              int j0=jao*nao;
              double Glk2[3*3];
              double Gjk[3]={ZERO,ZERO,ZERO};
#pragma unroll
              for (int kl=0;kl<3*3;kl++) Glk2[kl]=ZERO;
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
                int i0=iao*nao;
                int ij=i0+jao;
                double Gjl[3]={ZERO,ZERO,ZERO};
                double Gil[3]={ZERO,ZERO,ZERO};
		for (int k=0, kao=kao0; k<3; k++, kao++ ) {
                    int jk=jao*nao+kao;
                    int ik=iao*nao+kao;
                    double Gik=ZERO;
                    //double Gjk=ZERO;
		    for (int  l=0, lao=lao0; l<3; l++, lao++, ix++ ) {
			//if ( lao > kao ) continue;
			if ( lao <= kao && fabs(DSPP[ix]) > eps_eri ) {
			    double coe = ( kao == lao ? HALF : ONE );
			    double x, x4;
                            double dx;
			    //int ij, ik, il, jk, jl, kl, i0, j0;
			    int il, jl, kl;
			    x  = coe * DSPP[ix];

			    x4 = 4.e0 * x;
                            x *= -x_coef;
			    il = i0 + lao;
			    jl = j0 + lao;
			    kl = kao*nao + lao;
                            /*
			    G[ij] += x4*Ds[kl];
			    G[kl] += x4*Ds[ij];
			    G[ik] -=  x*Ds[jl];
			    G[il] -=  x*Ds[jk];
			    G[jk] -=  x*Ds[il];
			    G[jl] -=  x*Ds[ik];
                            */
                        Glk2[k*3+l] += x4*LDG(Ds[ij]);
                        Gij[i] += x4*LDG(Ds[kl]);
                        Gik    += x*LDG(Ds[jl]);
                        Gjk[k] += x*LDG(Ds[il]);
                        Gjl[l] += x*LDG(Ds[ik]);
                        Gil[l] += x*LDG(Ds[jk]);
			}
		    } // l
#ifdef USE_ATOMIC
                    atomicAdd(&G[ik], Gik);
#else
                    Gi[i*nao+kao] += Gik;
#endif
		} // k
#ifdef USE_ATOMIC
#pragma unroll
                for (int l=0,jl=j0+lao0;l<3;l++,jl++) atomicAdd(&G[jl],Gjl[l]);
#pragma unroll
                for (int l=0,il=i0+lao0;l<3;l++,il++) atomicAdd(&G[il],Gil[l]);
#else
#pragma unroll
                for (int l=0,lao=lao0;l<3;l++,lao++) Gj[lao] += Gjl[l];
#pragma unroll
                for (int l=0,il=i*nao+lao0;l<3;l++,il++) Gi[il] += Gil[l];
#endif /* USE_ATOMIC */
	    }	// for ( i, iao);
#ifdef USE_ATOMIC
#pragma unroll
            for (int k=0, kao=kao0; k<3; k++, kao++ ) 
#pragma unroll
              for (int l=0, lao=lao0; l<3; l++, lao++ ) 
                atomicAdd(&G[kao*nao+lao], Glk2[k*3+l]);
#pragma unroll
            for (int k=0, kao=kao0; k<3; k++, kao++ ) 
              atomicAdd(&G[j0+kao], +Gjk[k]);
#else
#pragma unroll
            for (int kl=0; kl<3*3; kl++)
              Gkl[klcs2*9+kl] += Glk2[kl];
#pragma unroll
            for (int k=0, kao=kao0; k<3; k++, kao++ ) 
              Gj[kao] += Gjk[k];
#endif /* USE_ATOMIC */
          } // block for additions
#ifdef DLB_KL_DSPP
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx] + tidw;
          } // while (iklcs)
#endif
	}	// for ( klcs );
//	klcs = klcs0;
#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
#if 0
#pragma unroll
        for (int i=0,ij=jao0*nao+iao0; i<6; i++,ij++)
          atomicAdd(&G[ij], Gij[i]);
#else
        {
          for (int i=0; i<6; i++) {
            double *ss = &shared[sOffset+tidx];
            __syncthreads();
            *ss = Gij[i];
            { // Reduce Gij within warp
              double *sGij = &shared[sOffset+widx*WARP_SIZE];
              warpReduce(sGij, tidw);
            }
            __syncthreads();
            if (tidx==0) { // Reduce Gij for warp masters
              double *sGij = &shared[sOffset];
              double dtmp = sGij[0];
              for (int j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
              atomicAdd(&G[jao0*nao+iao0+i], dtmp);
            }
          } // i
        }
#endif
        __syncthreads();
#else /* !USE_ATOMIC */
//#pragma unroll
        //for (int i=0; i<6; i++) Gj[iao0+i] += Gij[i];
        {
          int i, j;
          int nGi;
          int bs;
	  int i0 = iao0*nao;
	  int j0 = jao0*nao;
          int ijcs_next;
          int ics_next, jcs_next;
          for (i=0; i<6; i++) {
          double *ss = &shared[sOffset+tidx];
          __syncthreads();
          *ss = Gij[i];
          { // Reduce Gij within warp
            double *sGij = &shared[sOffset+widx*WARP_SIZE];
            warpReduce(sGij, tidw);
          }
          __syncthreads();
          if (tidx==0) { // Reduce Gij for warp masters
            double *sGij = &shared[sOffset];
            double dtmp = sGij[0];
            for (j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
            G[j0+iao0+i] += dtmp;
          }
          } // i
          ics_next=jcs_next=-1;
#ifdef GPU_DLB
          __syncthreads();
//          ijcs_next = ijcsw;
//          if (ijcs_next<ijcs1) { 
          ijcs_next = ijcs0 + ijcs1 - ijcsw - 1;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
//          if (ijcs_next>=ijcs0) { 
          if (ijcsw<ijcs1) { 
#else /* !GPU_DLB */
          ijcs_next = ijcs2+nwkblk;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
          if (ijcs_next<ijcs1) { 
#endif /* !GPU_DLB */
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
#ifndef GPU_DLB
          if (ics != ics_next || jcs != jcs_next) __syncthreads();
#endif
          nGi = num_Gi*2;
          bs = bidx*nthb*num_Gi;
          if (ics != ics_next) {
          for (int ii=tidx;ii<numao*6;ii+=nthb) {
            i = (ii/numao)*nao + ii%numao + minao;
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
          } // for i
          } // ics != ics_next
          if (jcs != jcs_next) {
//          for (i=tidx;i<nao;i+=nthb) {
          for (i=minao+tidx;i<=maxao;i+=nthb) {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao*6 + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          } // for i
          } // jcs != jcs_next
    __syncthreads();
        }
#endif /* !USE_ATOMIC */
    }		// for ( ijcs );

#ifndef USE_ATOMIC
    for (int i=tidx; i<(klcs1-klcs0)*3*3; i+=nthb) {
      int klcs, kl;
      int k, l;
      int kcs,lcs,kao,lao;
      klcs = i/9;
      kl = i - klcs*9;
      k = kl/3;
      l = kl-k*3;
      klcs += klcs0;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs] + k;
      lao = shel_ini[lcs] + l;
      G[kao*nao+lao] += Gkl[i];
    }
#endif

    return;
}

#ifdef DUMMY
/** (ds,ds)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_dsds_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, k, ix;
    int IJ, KL, ipat;
    int ijcs,        ijcs1;
    int klcs, klcs0;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao;
    int kcs, kat, kao, kao0, lcs, lat, lao;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DSDS[6*6];
    int ics2, jcs2;
    double dmax, dij;
    int Labcd = 9;
    int *p_ijcounter = &ijcounter[Labcd];
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao    = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<=ijcs; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao0   = shel_ini[kcs];
	    lao    = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_dsds_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DSDS );
	    ipat = ( (ics==kcs && jcs>lcs) ? true : false);
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		IJ = ( (iao*iao+iao)>>1) + jao;
		for ( k=0, kao=kao0; k<6; k++, kao++, ix++ ) {
		    KL = ( (kao*kao+kao)>>1) + lao;
		    if ( fabs(DSDS[ix]) <= eps_eri ) continue;
		    if ( IJ >= KL || ipat ) {
			double x, x4;
			int ij, ik, il, jk, jl, kl, i0, j0;
			coe = ONE;
			if ( iao==kao && jao==lao ) coe = HALF;
			x  = coe * DSDS[ix];
			/*
		// for check sum
		lsum += x;
		*/

			x4 = 4.e0 * x;
			i0 = iao*nao;
			j0 = jao*nao;
			ij = i0 + jao;
			ik = i0 + kao;
			il = i0 + lao;
			jk = j0 + kao;
			jl = j0 + lao;
			kl = kao*nao + lao;
			G[ij] += x4*Ds[kl];
			G[kl] += x4*Ds[ij];
			G[ik] -=  x*Ds[jl];
			G[il] -=  x*Ds[jk];
			G[jk] -=  x*Ds[il];
			G[jl] -=  x*Ds[ik];
		    }
		}	// for ( kao )
	    }		// for ( iao )
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[9] += lsum;
    */

    return 0;
}

/** (dp,ss)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_dpss_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, ix;
    int ijcs,        ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, lcs, lat, lao;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DPSS[6*3];
    int ics2, jcs2;
    double dmax, dij;
    int Labcd = 10;
    int *p_ijcounter = &ijcounter[Labcd];
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<klcs1; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao    = shel_ini[kcs];
	    lao    = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_dpss_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DPSS );
	    coe = ( kao == lao ? HALF : ONE );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		for ( j=0, jao=jao0; j<3; j++, jao++, ix++ ) {
		    if ( fabs(DPSS[ix]) > eps_eri ) {
			double x, x4;
			int ij, ik, il, jk, jl, kl, i0, j0;
			x  = coe * DPSS[ix];
			/*
		// for check sum
		lsum += x;
		*/

			x4 = 4.e0 * x;
			i0 = iao*nao;
			j0 = jao*nao;
			ij = i0 + jao;
			ik = i0 + kao;
			il = i0 + lao;
			jk = j0 + kao;
			jl = j0 + lao;
			kl = kao*nao + lao;
			G[ij] += x4*Ds[kl];
			G[kl] += x4*Ds[ij];
			G[ik] -=  x*Ds[jl];
			G[il] -=  x*Ds[jk];
			G[jk] -=  x*Ds[il];
			G[jl] -=  x*Ds[ik];
		    }
		}
	    }	// for ( i, iao);
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[10] += lsum;
    */

    return 0;
}
#endif

/** (dp,ps)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
NREGS_255
gpu_twoint_direct_dpps_( const int nwks, const int iwk,
   const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int La=2, Lb=1, Lc=1, Ld=0;
    int Lab, Lcd;
    int Labcd = 11;
    int *p_ijcounter = &ijcounter[Labcd];
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, klcs0a;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao0, jcs, jat, jao0;
    int kcs, kat, kao0, lcs, lat, lao0;
    double DPPS[6*3*3];
#ifdef USE_INSTANT_SCHWARZ
    float val_ab;
#endif
    int ics2, jcs2;
    int nwkblk, iwkblk;
    double *G, *Gkl;
    double *Gi, *Gj;
    double *work;
    int mincs,maxcs,minao,maxao;
    int minao1[2], maxao1[2];
    int numao;
    int *sklcs;
    __shared__ int nklcs;
    __shared__ int ijcsw;
    __shared__ int nijcsw;
#ifdef CUDA_FMT_M_SM
    size_t tsize = FMT_m_size[4];
    double *tbl = FMT_m_table4;
    for (int i=tidx; i<tsize; i+=nthb) shared[i] = tbl[i];
    __syncthreads();
    double Gij[3];
    size_t sOffset = tsize;
    tsize += nthb;
#else
    double *Gij = &shared[tidx*3];
    size_t tsize = nthb*3;
    size_t sOffset = 0;
#endif
#ifdef DLB_KL_DSPS
//    __shared__ volatile int klcsw[NTHREADS/WARP_SIZE];
//    volatile int *klcsw = (int *)&shared[nthb*3];
    volatile int *klcsw = (int *)&shared[tsize];
    __shared__ int cklcs;
#endif
    __shared__ double A[3];
    __shared__ double BA[3];

    minao1[0] = shel_ini[leading_cs[La]];
    minao1[1] = shel_ini[leading_cs[Lb]];
    minao = MIN2(minao1[0], minao1[1]);
    maxao1[0] = shel_ini[leading_cs[La+1]] - 1;
    maxao1[1] = shel_ini[leading_cs[Lb+1]] - 1;
    maxao = MAX2(maxao1[0], maxao1[1]);
#ifdef ADD_FULL_NAO
    minao=0;
    maxao=nao-1;
#endif
    numao = maxao - minao + 1;

    sklcs = sklcs_b + bidx * max_num_klcs;
    G   = G_b   + bidx * nao * nao;
    Gkl = Gkl_b + bidx * num_Gkl;
    Gi   = Gi_t + (bidx * nthb + tidx) * num_Gi;
    Gj   = Gi + nao*3;
    work = work_b + bidx * nthb * WORK_SIZE;

//    nwkblk = nworkers * ndev * nblk;
//    iwkblk = workerid * ndev * nblk + idev * nblk + bidx;
//    nwkblk = nwks;
//    iwkblk = iwk + bidx;
    nwkblk = nwks * nblk;
    iwkblk = iwk * nblk + bidx;

    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lcd];
    ijcs1 = leading_cs_pair[Lcd+1];
    klcs0 = leading_cs_pair[Lab];
    klcs1 = leading_cs_pair[Lab+1];

#ifndef USE_ATOMIC
    for (int idx=0; idx<nthb; idx++) {
      double *Gi0   = Gi_t + (bidx * nthb + idx) * num_Gi;
      for (int i=tidx; i<nao*4; i+=nthb ) Gi0[i]=ZERO;
    }
    for (int i=tidx; i<(klcs1-klcs0)*6*3; i+=nthb) Gkl[i]=ZERO;
    __syncthreads();
#endif

#ifdef GPU_DLB
//    if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
//    ijcsw = ijcs0 + iwk + nwks * bidx;
//    nijcsw = 0;
    if (tidx==0) {
      ijcsw = ijcs0 + iwk + nwks * bidx * NIJCSW;
      nijcsw = NIJCSW-1;
    }
    __syncthreads();
#endif

//    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
#ifdef GPU_DLB
    while(ijcsw<ijcs1) {
#else
    for (int ijcs2=ijcs0+iwkblk; ijcs2<ijcs1; ijcs2+=nwkblk ) {
#endif
#ifdef GPU_DLB
//        ijcs = ijcsw;
        ijcs = ijcs0 + ijcs1 - ijcsw - 1;
        /*
        __syncthreads();
        if (tidx==0) ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,1)*nwks;
        */
#else
      ijcs = ijcs2;
#endif
#ifdef SORT_IJ_SCHWARZ
      ijcs = sorted_csp[ijcs];
#endif
#ifdef DLB_KL_DPPS
        if(tidx==0) cklcs=nthb;
        __syncthreads();
#endif
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
//        if(tidx==0) printf("%d:0 %d %d %d\n",tidx,ijcs,ics,jcs);
#ifdef USE_INSTANT_SCHWARZ
	val_ab = csp_schwarz[ijcs];
#else /* !USE_INSTANT_SCHWARZ */
        {
	  float val_ab = csp_schwarz[ijcs];
//          int sklcs_l[NKLBUF];
          __syncthreads();
          if(tidx == 0) nklcs = 0;
          __syncthreads();
	  for (int klcs2 = klcs0+tidx ; klcs2<klcs1; klcs2+=nthb) {
//	  for (int klcs2 = klcs1-tidx-1; klcs2>=klcs0 ; klcs2-=nthb) {
              int klcs=klcs2;
              int kcs, lcs;
              int iklcs;
              float dmax;
	      float val_abcd;
#ifdef SORT_KL_SCHWARZ
          klcs = sorted_csp[klcs2];
#endif
	      val_abcd = val_ab * csp_schwarz[klcs];
	      if ( val_abcd < eps_ps4 ) continue;
	      kcs    = csp_ics[klcs];
	      lcs    = csp_jcs[klcs];
              dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	      if ( dmax*val_abcd < eps_sch ) continue;
//              iklcs = atomicInc(&nklcs, max_num_klcs);
              iklcs = atomicAdd(&nklcs, 1);
#ifdef SORT_KL_SCHWARZ
              klcs = klcs2;
#endif
              sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
//          if(bidx==0&&tidx==0) printf("nklcs = %d\n", nklcs);
        }
#endif /* USE_INSTANT_SCHWARZ */
#pragma unroll
        for (int i=0;i<3;i++) Gij[i] = ZERO;
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
        __syncthreads();
        if(tidx == 0) {
        /*
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        */
        BA[0] = LDG(atom_x[jat]) + (A[0] = -LDG(atom_x[iat]));
        BA[1] = LDG(atom_y[jat]) + (A[1] = -LDG(atom_y[iat]));
        BA[2] = LDG(atom_z[jat]) + (A[2] = -LDG(atom_z[iat]));
        }
        __threadfence_block();
        __syncthreads();

//        if(tidx==0) printf("%d:k %d %d\n",tidx,klcs0,klcs1);
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
	
//	for (klcs = klcs0 ; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (klcs = klcs0+tidx ; klcs<klcs1; klcs+=nthb ) {
#ifdef DLB_KL_DPPS
  NOT supported combination of defines.
#endif
#else /* !USE_INSTANT_SCHWARZ */
#ifndef DLB_KL_DPPS
//	for (int iklcs = tidx ; iklcs<nklcs; iklcs+=nthb ) {
	for (int iklcs2 = tidx ; iklcs2<nklcs; iklcs2+=nthb ) {
          int iklcs;
          int it = iklcs2 / nthb;
          if (it%2 == 0) iklcs = iklcs2;
          else {
            int ilast = MIN2(nklcs, (it+1)*nthb);
            iklcs = it*nthb + ilast - iklcs2 - 1;
          }
#else /* !DLB_KL_DPPS */
        {
          int iklcs;
          if (tidw==0) klcsw[widx] = widx * warpSize;
          iklcs = klcsw[widx] + tidw;
        while(iklcs<nklcs) {
#endif /* !DLB_KL_DPPS */
#endif /* !USE_INSTANT_SCHWARZ */
            double DC[3], AC[3];
            float dmax;
            int klcs2;
#ifdef USE_INSTANT_SCHWARZ
	    float val_abcd = val_ab * csp_schwarz[klcs];
	    if ( val_abcd < eps_ps4 ) continue;
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
            dmax = gpu_dmax6(ics,jcs,kcs,lcs);
	    if ( dmax*val_abcd < eps_sch ) continue;
#else
            klcs = sklcs[iklcs];
#ifdef SORT_KL_SCHWARZ
            klcs = sorted_csp[klcs];
#endif
#endif /* USE_INSTANT_SCHWARZ */
            klcs2 = klcs - klcs0;
//         if (tidx==0) printf("%d:kl %d %d\n",tidx,klcs, klcs2);
	    kcs    = LDG(csp_ics[klcs]);
	    lcs    = LDG(csp_jcs[klcs]);
	    klps0  = LDG(csp_leading_ps_pair[klcs]);
	    nklps  = LDG(csp_leading_ps_pair[klcs+1])-klps0;
	    kat    = LDG(shel_atm[kcs]);
	    lat    = LDG(shel_atm[lcs]);
	    kao0   = LDG(shel_ini[kcs]);
	    lao0   = LDG(shel_ini[lcs]);
            /*
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
            */
            {
              double dtmp;
              AC[0] = A[0] + (dtmp = LDG(atom_x[kat]));
              DC[0] = LDG(atom_x[lat]) - dtmp;
              AC[1] = A[1] + (dtmp = LDG(atom_y[kat]));
              DC[1] = LDG(atom_y[lat]) - dtmp;
              AC[2] = A[2] + (dtmp = LDG(atom_z[kat]));
              DC[2] = LDG(atom_z[lat]) - dtmp;
            }
/*
	    gpu_twoint_core_dpps_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DPPS );
*/
// (ps,dp) -> (dp,ps)
	    //gpu_twoint_core_dpps_(
	    gpu_twoint_core_os_dpps(
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,   AC,      DPPS);
            {
              int jao = jao0;
              int j0 = jao*nao;
              int ix = 0;
              double Gjl[3] = {ZERO,ZERO,ZERO};
              double Gil[9];
#pragma unroll
              for (int i=0;i<9;i++) Gil[i]=ZERO;
#pragma unroll
	    for (int k=0, kao=kao0, ix=0; k<6; k++, kao++ ) {
              int jk = j0 + kao;
              double Gkl2[3] = {ZERO,ZERO,ZERO};
              double Gjk = ZERO;
              double Gik[3] = {ZERO,ZERO,ZERO};
#pragma unroll
		for (int l=0, lao=lao0; l<3; l++, lao++ ) {
                  int jl = j0 + lao;
                  int kl = kao*nao+lao;
#pragma unroll
		    for (int i=0, iao=iao0; i<3; i++, iao++, ix++ ) {
                      double x = DPPS[ix];
			if ( fabs(x) > eps_eri ) {
			    int ij, ik, il, i0;
                            double dx, x4;

			    x4 = 4.e0 * x;
                            x *= -x_coef;
			    i0 = iao*nao;
			    ij = i0 + jao;
			    ik = i0 + kao;
			    il = i0 + lao;
                            /*
			    G[ij] += x4*Ds[kl];
			    G[kl] += x4*Ds[ij];
			    G[ik] -=  x*Ds[jl];
			    G[il] -=  x*Ds[jk];
			    G[jk] -=  x*Ds[il];
			    G[jl] -=  x*Ds[ik];
                            */
			
                          Gij[i]  += x4*LDG(Ds[kl]);
                          Gkl2[l] += x4*LDG(Ds[ij]);
                          Gik[i]     += x*LDG(Ds[jl]);
                          Gil[i*3+l] += x*LDG(Ds[jk]);
                          Gjk        += x*LDG(Ds[il]);
                          Gjl[l]     += x*LDG(Ds[ik]);
			}
		    } // i
		} // l
#ifdef USE_ATOMIC
                atomicAdd(&G[j0+kao], Gjk);
#pragma unroll
                for (int i=0,iao=iao0; i<3; i++,iao++)
                  atomicAdd(&G[iao*nao+kao], Gik[i]);
#pragma unroll
                for (int l=0,lao=lao0; l<3; l++,lao++)
                  atomicAdd(&G[kao*nao+lao], Gkl2[l]);
#else
                Gj[kao] += Gjk;
                for (int i=0; i<3; i++)
                  Gi[i*nao+kao] += Gik[i];
#pragma unroll
                for (int l=0; l<3; l++)
                  Gkl[klcs2*6*3+k*3+l] += Gkl2[l];
#endif
	    }	// for ( k, kao);
#ifdef USE_ATOMIC
#pragma unroll
            for (int i=0,iao=iao0; i<3; i++,iao++)
#pragma unroll
              for (int l=0,lao=lao0; l<3; l++,lao++)
                atomicAdd(&G[iao*nao+lao], Gil[i*3+l]);
#pragma unroll
            for (int l=0,lao=lao0; l<3; l++,lao++)
              atomicAdd(&G[j0+lao], Gjl[l]);
#else
#pragma unroll
            for (int i=0,iao=iao0; i<3; i++,iao++)
#pragma unroll
              for (int l=0,lao=lao0; l<3; l++,lao++)
                Gi[i*nao+lao] += Gil[i*3+l];
#pragma unroll
            for (int l=0,lao=lao0; l<3; l++,lao++)
              Gj[lao] += Gjl[l];
#endif
          } // block for additions
#ifdef DLB_KL_DPPS
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx] + tidw;
          } // while (iklcs)
#endif
	}	// for ( klcs );
//	klcs = klcs0;
#ifdef GPU_DLB
        if (tidx==0) {
          if (nijcsw > 0) {
            ijcsw += nwks;
            nijcsw--;
          } else {
            ijcsw = ijcs0 + iwk + atomicAdd(p_ijcounter,NIJCSW)*nwks;
            nijcsw = NIJCSW-1;
          }
        }
#endif
#ifdef USE_ATOMIC
#if 0
#pragma unroll
        for (int i=0; i<3; i++)
          atomicAdd(&G[jao0*nao+iao0+i], Gij[i]);
#else
        {
#ifndef CUDA_FMT_M_SM
          double tGij[3];
          for (int i=0; i<3; i++) tGij[i] = Gij[i];
#else
          double *tGij = Gij;
#endif
          for (int i=0; i<3; i++) {
            double *ss = &shared[sOffset+tidx];
            __syncthreads();
            *ss = tGij[i];
            { // Reduce Gij within warp
              double *sGij = &shared[sOffset+widx*WARP_SIZE];
              warpReduce(sGij, tidw);
            }
            __syncthreads();
            if (tidx==0) { // Reduce Gij for warp masters
              double *sGij = &shared[sOffset];
              double dtmp = sGij[0];
              for (int j=1; j<nwrp; j++) dtmp += sGij[j*WARP_SIZE];
              atomicAdd(&G[jao0*nao+iao0+i], dtmp);
            }
          } // i
        }
#endif
        __syncthreads();
#else /* !USE_ATOMIC */
#pragma unroll
        for (int i=0; i<3; i++) Gj[iao0+i] += Gij[i];
        {
          int i,j;
          int nGi;
          int bs;
	  int i0 = iao0*nao;
	  int j0 = jao0*nao;
          int ijcs_next;
          int ics_next, jcs_next;
          ics_next=jcs_next=-1;
#ifdef GPU_DLB
          __syncthreads();
//          ijcs_next = ijcsw;
//          if (ijcs_next<ijcs1) { 
          ijcs_next = ijcs0 + ijcs1 - ijcsw - 1;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
//          if (ijcs_next>=ijcs0) { 
          if (ijcsw<ijcs1) { 
#else /* !GPU_DLB */
          ijcs_next = ijcs2+nwkblk;
#ifdef SORT_IJ_SCHWARZ
          ijcs_next = sorted_csp[ijcs_next];
#endif
          if (ijcs_next<ijcs1) { 
#endif /* !GPU_DLB */
	    ics_next = csp_ics[ijcs_next];
	    jcs_next = csp_jcs[ijcs_next];
          }
#ifndef GPU_DLB
          if (ics != ics_next || jcs != jcs_next) __syncthreads();
#endif
          nGi = num_Gi*2;
          bs = bidx*nthb*num_Gi;
          if (ics != ics_next) {
          for (int ii=tidx;ii<numao*3;ii+=nthb) {
            i = (ii/numao)*nao + ii%numao + minao;
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[i0+i] += Gtmp;
          } // for i
          } // ics != ics_next
          if (jcs != jcs_next) {
//          for (i=tidx;i<nao;i+=nthb) {
          for (i=minao+tidx;i<=maxao;i+=nthb) {
            double Gtmp = ZERO;
            double *pG = Gi_t + bs + nao*3 + i;
            double *pG1 = pG + num_Gi;
#pragma unroll
            for (j=0;j<nthb/2;j++) {
              Gtmp += (*pG + *pG1);
              *pG = *pG1 = ZERO;
              pG  += nGi;
              pG1 += nGi;
            }
            if (j*2!=nthb) {
              Gtmp += *pG;
              *pG = ZERO;
            }
            G[j0+i] += Gtmp;
          } // for i
          } // jcs != jcs_next
    __syncthreads();
        }
#endif /* !USE_ATOMIC */
    }		// for ( ijcs );

#ifndef USE_ATOMIC
    for (int i=tidx; i<(klcs1-klcs0)*6*3; i+=nthb) {
      int klcs, kl;
      int k, l;
      int kcs,lcs,kao,lao;
      klcs = i/18;
      kl = i - klcs*18;
      k = kl/3;
      l = kl-k*3;
      klcs += klcs0;
      kcs = csp_ics[klcs];
      lcs = csp_jcs[klcs];
      kao = shel_ini[kcs] + k;
      lao = shel_ini[lcs] + l;
      G[kao*nao+lao] += Gkl[i];
    }
#endif

    return;
}

#ifdef DUMMY
/** (dp,pp)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_dppp_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, k, l, ix;
    int ijcs,        ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DPPP[6*3*3*3];
    int ics2, jcs2;
    double dmax, dij;
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<klcs1; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao0   = shel_ini[kcs];
	    lao0   = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_dppp_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DPPP );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		for ( j=0, jao=jao0; j<3; j++, jao++ ) {
		    for (k=0, kao=kao0; k<3; k++, kao++ ) {
			for ( l=0, lao=lao0; l<3; l++, lao++, ix++ ) {
			    if ( lao > kao ) continue;
			    coe = ( kao == lao ? HALF : ONE );
			    if ( fabs(DPPP[ix]) > eps_eri ) {
				double x, x4;
				int ij, ik, il, jk, jl, kl, i0, j0;
				x  = coe * DPPP[ix];
				/*
		// for check sum
		lsum += x;
		*/

				x4 = 4.e0 * x;
				i0 = iao*nao;
				j0 = jao*nao;
				ij = i0 + jao;
				ik = i0 + kao;
				il = i0 + lao;
				jk = j0 + kao;
				jl = j0 + lao;
				kl = kao*nao + lao;
				G[ij] += x4*Ds[kl];
				G[kl] += x4*Ds[ij];
				G[ik] -=  x*Ds[jl];
				G[il] -=  x*Ds[jk];
				G[jk] -=  x*Ds[il];
				G[jl] -=  x*Ds[ik];
			    }
			}
		    }
		}
	    }	// for ( i, iao);
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[12] += lsum;
    */

    return 0;
}

/** (dp,ds)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_dpds_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, k, ix;
    int ijcs,        ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd;
    double DPDS[6*3*6];
    int ics2, jcs2;
    double dmax, dij;
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<klcs1; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao0   = shel_ini[kcs];
	    lao    = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_dpds_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DPDS );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		for ( j=0, jao=jao0; j<3; j++, jao++ ) {
		    for (k=0, kao=kao0; k<6; k++, kao++, ix++ ) {
			if ( fabs(DPDS[ix]) > eps_eri ) {
			    double x, x4;
			    int ij, ik, il, jk, jl, kl, i0, j0;
			    x  = DPDS[ix];
			    /*
		// for check sum
		lsum += x;
		*/

			    x4 = 4.e0 * x;
			    i0 = iao*nao;
			    j0 = jao*nao;
			    ij = i0 + jao;
			    ik = i0 + kao;
			    il = i0 + lao;
			    jk = j0 + kao;
			    jl = j0 + lao;
			    kl = kao*nao + lao;
			    G[ij] += x4*Ds[kl];
			    G[kl] += x4*Ds[ij];
			    G[ik] -=  x*Ds[jl];
			    G[il] -=  x*Ds[jk];
			    G[jk] -=  x*Ds[il];
			    G[jl] -=  x*Ds[ik];
			}
		    }
		}
	    }	// for ( i, iao);
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[13] += lsum;
    */

    return 0;
}

/** (dp,dp)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_dpdp_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, k, l, ix, ipat;
    int IJ, KL;
    int ijcs,        ijcs1;
    int klcs, klcs0;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DPDP[6*3*6*3];
    int ics2, jcs2;
    double dmax, dij;
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<=ijcs; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao0   = shel_ini[kcs];
	    lao0   = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_dpdp_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DPDP );
	    ipat = ( (ics==kcs && jcs>lcs) ? true : false );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		for ( j=0, jao=jao0; j<3; j++, jao++ ) {
		    IJ = ((iao*iao+iao)>>1) + jao;
		    for ( k=0, kao=kao0; k<6; k++, kao++ ) {
			for ( l=0, lao=lao0; l<3; l++, lao++, ix++ ) {
			    KL = ((kao*kao+kao)>>1) + lao;
			    if ( fabs(DPDP[ix]) > eps_eri ) {
				if ( IJ>=KL || ipat ) {
				    double x, x4;
				    int ij, ik, il, jk, jl, kl, i0, j0;
				    coe = ( IJ==KL ? HALF : ONE );
				    x  = coe * DPDP[ix];
				    /*
		// for check sum
		lsum += x;
		*/

				    x4 = 4.e0 * x;
				    i0 = iao*nao;
				    j0 = jao*nao;
				    ij = i0 + jao;
				    ik = i0 + kao;
				    il = i0 + lao;
				    jk = j0 + kao;
				    jl = j0 + lao;
				    kl = kao*nao + lao;
				    G[ij] += x4*Ds[kl];
				    G[kl] += x4*Ds[ij];
				    G[ik] -=  x*Ds[jl];
				    G[il] -=  x*Ds[jk];
				    G[jk] -=  x*Ds[il];
				    G[jl] -=  x*Ds[ik];
				}
			    }
			}
		    }	// for ( kao )
		}
	    }		// for ( iao )
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[14] += lsum;
    */

    return 0;
}

/** (dd,ss)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_ddss_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, ix;
    int ijcs,        ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, lcs, lat, lao;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe, coe0;
    double DDSS[6*6];
    int ics2, jcs2;
    double dmax, dij;
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<klcs1; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao    = shel_ini[kcs];
	    lao    = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_ddss_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DDSS );
	    coe0 = ( kao == lao ? HALF : ONE );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		for ( j=0, jao=jao0; j<6; j++, jao++, ix++ ) {
		    if ( jao > iao ) continue;
		    if ( fabs(DDSS[ix]) > eps_eri ) {
			double x, x4;
			int ij, ik, il, jk, jl, kl, i0, j0;
			coe = coe0;
			if ( iao == jao ) coe *= HALF;
			x  = coe * DDSS[ix];
			/*
		// for check sum
		lsum += x;
		*/

			x4 = 4.e0 * x;
			i0 = iao*nao;
			j0 = jao*nao;
			ij = i0 + jao;
			ik = i0 + kao;
			il = i0 + lao;
			jk = j0 + kao;
			jl = j0 + lao;
			kl = kao*nao + lao;
			G[ij] += x4*Ds[kl];
			G[kl] += x4*Ds[ij];
			G[ik] -=  x*Ds[jl];
			G[il] -=  x*Ds[jk];
			G[jk] -=  x*Ds[il];
			G[jl] -=  x*Ds[ik];
		    }
		} // for ( jao );
	    }	// for ( iao );
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[15] += lsum;
    */

    return 0;
}

/** (dd,ps)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_ddps_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, k, ix;
    int ijcs,        ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DDPS[6*6*3];
    int ics2, jcs2;
    double dmax, dij;
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<klcs1; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao0   = shel_ini[kcs];
	    lao    = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_ddps_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DDPS );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix += 3; continue; }
		    coe = ( iao==jao ? HALF : ONE );
		    for ( k=0, kao=kao0; k<3; k++, kao++, ix++ ) {
			if ( fabs(DDPS[ix]) > eps_eri ) {
			    double x, x4;
			    int ij, ik, il, jk, jl, kl, i0, j0;
			    x  = coe * DDPS[ix];
			    /*
		// for check sum
		lsum += x;
		*/

			    x4 = 4.e0 * x;
			    i0 = iao*nao;
			    j0 = jao*nao;
			    ij = i0 + jao;
			    ik = i0 + kao;
			    il = i0 + lao;
			    jk = j0 + kao;
			    jl = j0 + lao;
			    kl = kao*nao + lao;
			    G[ij] += x4*Ds[kl];
			    G[kl] += x4*Ds[ij];
			    G[ik] -=  x*Ds[jl];
			    G[il] -=  x*Ds[jk];
			    G[jk] -=  x*Ds[il];
			    G[jl] -=  x*Ds[ik];
			}	// if ( fabs );
		    }	// for ( kao )
		}	// for ( jao )
	    }	// for ( iao )
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[16] += lsum;
    */

    return 0;
}

/** (dd,pp)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_ddpp_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, k, l, ix;
    int ijcs,        ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe, coe0;
    double DDPP[6*6*3*3];
    int ics2, jcs2;
    double dmax, dij;
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<klcs1; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao0   = shel_ini[kcs];
	    lao0   = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_ddpp_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DDPP );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix+=3*3; continue; }
		    coe0 = (iao==jao ? HALF : ONE );
		    for (k=0, kao=kao0; k<3; k++, kao++ ) {
			for ( l=0, lao=lao0; l<3; l++, lao++, ix++ ) {
			    if ( lao > kao ) continue;
			    coe = coe0 * ( kao==lao ? HALF : ONE );
			    if ( fabs(DDPP[ix]) > eps_eri ) {
				double x, x4;
				int ij, ik, il, jk, jl, kl, i0, j0;
				x  = coe * DDPP[ix];
				/*
		// for check sum
		lsum += x;
		*/

				x4 = 4.e0 * x;
				i0 = iao*nao;
				j0 = jao*nao;
				ij = i0 + jao;
				ik = i0 + kao;
				il = i0 + lao;
				jk = j0 + kao;
				jl = j0 + lao;
				kl = kao*nao + lao;
				G[ij] += x4*Ds[kl];
				G[kl] += x4*Ds[ij];
				G[ik] -=  x*Ds[jl];
				G[il] -=  x*Ds[jk];
				G[jk] -=  x*Ds[il];
				G[jl] -=  x*Ds[ik];
			    }
			}
		    }
		}
	    }	// for ( i, iao);
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[17] += lsum;
    */

    return 0;
}

/** (dd,ds)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_ddds_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, k, ix;
    int ijcs,        ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DDDS[6*6*6];
    int ics2, jcs2;
    double dmax, dij;
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<klcs1; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao0   = shel_ini[kcs];
	    lao    = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_ddds_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DDDS );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix += 6; continue; }
		    coe = ( iao==jao ? HALF : ONE );
		    for ( k=0, kao=kao0; k<6; k++, kao++, ix++ ) {
			if ( fabs(DDDS[ix]) > eps_eri ) {
			    double x, x4;
			    int ij, ik, il, jk, jl, kl, i0, j0;
			    x  = coe * DDDS[ix];
			    /*
		// for check sum
		lsum += x;
		*/

			    x4 = 4.e0 * x;
			    i0 = iao*nao;
			    j0 = jao*nao;
			    ij = i0 + jao;
			    ik = i0 + kao;
			    il = i0 + lao;
			    jk = j0 + kao;
			    jl = j0 + lao;
			    kl = kao*nao + lao;
			    G[ij] += x4*Ds[kl];
			    G[kl] += x4*Ds[ij];
			    G[ik] -=  x*Ds[jl];
			    G[il] -=  x*Ds[jk];
			    G[jk] -=  x*Ds[il];
			    G[jl] -=  x*Ds[ik];
			}	// if ( fabs );
		    }	// for ( kao )
		}	// for ( jao )
	    }	// for ( iao )
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[18] += lsum;
    */

    return 0;
}

/** (dd,dp)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_dddp_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, k, l, ix;
    int ijcs,        ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DDDP[6*6*6*3];
    int ics2, jcs2;
    double dmax, dij;
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<klcs1; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao0   = shel_ini[kcs];
	    lao0   = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_dddp_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DDDP );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix += 6*3; continue; }
		    coe = ( iao==jao ? HALF : ONE );
		    for ( k=0, kao=kao0; k<6; k++, kao++ ) {
			for ( l=0, lao=lao0; l<3; l++, lao++, ix++ ) {
			    if ( fabs(DDDP[ix]) > eps_eri ) {
				double x, x4;
				int ij, ik, il, jk, jl, kl, i0, j0;
				x  = coe * DDDP[ix];
				/*
		// for check sum
		lsum += x;
		*/

				x4 = 4.e0 * x;
				i0 = iao*nao;
				j0 = jao*nao;
				ij = i0 + jao;
				ik = i0 + kao;
				il = i0 + lao;
				jk = j0 + kao;
				jl = j0 + lao;
				kl = kao*nao + lao;
				G[ij] += x4*Ds[kl];
				G[kl] += x4*Ds[ij];
				G[ik] -=  x*Ds[jl];
				G[il] -=  x*Ds[jk];
				G[jk] -=  x*Ds[il];
				G[jl] -=  x*Ds[ik];
			    }	// if ( fabs );
			}
		    }	// for ( kao )
		}	// for ( jao )
	    }	// for ( iao )
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[19] += lsum;
    */

    return 0;
}

/** (dd,dd)タイプの積分計算、および、G行列計算を行う関数
 *
 * @ingroup integ-twoint-direct
 * */
__global__ void
#ifdef MY_KERNEL_MIN_BLOCKS
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
#endif
gpu_twoint_direct_dddd_(
	// paralleization
	const int nworkers, const int workerid,
	// integral type data
	const int La, const int Lb, const int Lc, const int Ld,
	// basis set & cutoff table data
	const int shel_atm[], const int shel_ini[],
	const double atom_x[], const double atom_y[],
	const double atom_z[], const int leading_cs_pair[],
	const double csp_schwarz[],
	const int csp_ics[], const int csp_jcs[],
	const int csp_leading_ps_pair[],
	const double psp_zeta[], const double psp_dkps[],
	const double psp_xiza[],
	// concerned about buffered direct method
	const int last_ijcs, const int last_klcs,
	// density matrix & G-matrix data
	const int nao, const double Ds[], double G[],
	const int ncs, const float Dcs[] ) {
    int Lab, Lcd, i, j, k, l, ix, ipat;
    int I2, IJ, K2, KL;
    int ijcs,        ijcs1;
    int klcs, klcs0;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe, coe0;
    double DDDD[6*6*6*6];
    int ics2, jcs2;
    double dmax, dij;
    /*
    // for check sum
    double lsum = ZERO;
    */


    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    if ( last_ijcs != -1 ) {
	ijcs = last_ijcs;
	klcs = last_klcs+1;
    } else {
	ijcs = leading_cs_pair[Lab] + workerid;
	klcs = klcs0;
    }

    for (  ; ijcs<ijcs1; ijcs+=nworkers ) {
	val_ab = csp_schwarz[ijcs];
	ics    = csp_ics[ijcs];
	jcs    = csp_jcs[ijcs];
	ijps0  = csp_leading_ps_pair[ijcs];
	nijps  = csp_leading_ps_pair[ijcs+1]-ijps0;
	iat    = shel_atm[ics];
	jat    = shel_atm[jcs];
	iao0   = shel_ini[ics];
	jao0   = shel_ini[jcs];
	A[0]=atom_x[iat]; A[1]=atom_y[iat]; A[2]=atom_z[iat];
	B[0]=atom_x[jat]; B[1]=atom_y[jat]; B[2]=atom_z[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        ics2 = ics*ncs;
        jcs2 = jcs*ncs;
        dij = Dcs[ics2+jcs];
	
	for ( ; klcs<=ijcs; klcs++ ) {
	    kcs    = csp_ics[klcs];
	    lcs    = csp_jcs[klcs];
          dmax = MAX2(dij, Dcs[kcs*ncs+lcs]) * 4.0e0;
          dmax = MAX2(dmax, Dcs[ics2+kcs]);
          dmax = MAX2(dmax, Dcs[ics2+lcs]);
          dmax = MAX2(dmax, Dcs[jcs2+kcs]);
          dmax = MAX2(dmax, Dcs[jcs2+lcs]);
	    val_cd = csp_schwarz[klcs];
//	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    if ( dmax*val_ab*val_cd < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair[klcs];
	    nklps  = csp_leading_ps_pair[klcs+1]-klps0;
	    kat    = shel_atm[kcs];
	    lat    = shel_atm[lcs];
	    kao0   = shel_ini[kcs];
	    lao0   = shel_ini[lcs];
	    C[0]=atom_x[kat]; C[1]=atom_y[kat]; C[2]=atom_z[kat];
	    D[0]=atom_x[lat]; D[1]=atom_y[lat]; D[2]=atom_z[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    twoint_core_dddd_(
		    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
		    &psp_xiza[ijps0], BA,
		    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
		    &psp_xiza[klps0], DC,   AC,      DDDD );
	    ipat = ( (ics==kcs && jcs>lcs) ? true : false );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		I2 = (iao*iao+iao)>>1;
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix+=6*6; continue; }
		    IJ = I2 + jao;
		    coe0 = ( iao==jao ? HALF : ONE );
		    for ( k=0, kao=kao0; k<6; k++, kao++ ) {
			K2 = (kao*kao+kao)>>1;
			for ( l=0, lao=lao0; l<6; l++, lao++, ix++ ) {
			    if ( lao>kao ) continue;
			    if ( fabs(DDDD[ix]) > eps_eri ) {
				double x, x4;
				int ij, ik, il, jk, jl, kl, i0, j0;
				KL = K2 + lao;
				if ( IJ >= KL || ipat ) {
				    coe = coe0;
				    if ( kao==lao ) coe *= HALF;
				    if ( KL == IJ ) coe *= HALF;
				    x  = coe * DDDD[ix];
				    /*
		// for check sum
		lsum += x;
		*/

				    x4 = 4.e0 * x;
				    i0 = iao*nao;
				    j0 = jao*nao;
				    ij = i0 + jao;
				    ik = i0 + kao;
				    il = i0 + lao;
				    jk = j0 + kao;
				    jl = j0 + lao;
				    kl = kao*nao + lao;
				    G[ij] += x4*Ds[kl];
				    G[kl] += x4*Ds[ij];
				    G[ik] -=  x*Ds[jl];
				    G[il] -=  x*Ds[jk];
				    G[jk] -=  x*Ds[il];
				    G[jl] -=  x*Ds[ik];
				}
			    } // if ( fabs )
			} // for ( lao )
		    } // for ( kao )
		} // for ( jao )
	    }	// for ( iao )
	}	// for ( klcs );
	klcs = klcs0;
    }		// for ( ijcs );
    /*
    // for check sum
#pragma omp atomic
    check_sum[20] += lsum;
    */

    return 0;
}
#endif /* DUMMY */

#undef WORK

#undef NREGS_064
#undef NREGS_128
#undef NREGS_255

