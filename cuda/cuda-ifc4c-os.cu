#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "ofmo-def.h"
#ifdef __cplusplus
}
#endif

#include "cudalib.h"
#include "cuda-twoint-core.h"
#include "cuda-twoint-core-os.h"
#include "cuda-utils.h"
#ifdef CUDA_FMT_M
#include "cuda-fmt-m.h"
#endif

#ifndef ZERO
#define ZERO    0.e0
#endif
#ifndef ONE
#define ONE     1.e0
#endif
#ifndef TWO
#define TWO     2.e0
#endif
#ifndef HALF
#define HALF    0.5e0
#endif

//#define EPS_ERI 1.e-15
//#define EPS_PS4 1.e-30
//#if CUDA_ARCH >= 350
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
/*
#define NREGS_064 __launch_bounds__(256,4)
#define NREGS_128 __launch_bounds__(256,2)
#define NREGS_255 __launch_bounds__(256,1)
*/
#define NREGS_0 __launch_bounds__(256)
#define NREGS_1 __launch_bounds__(256,1)
#define NREGS_2 __launch_bounds__(256,2)
#define NREGS_3 __launch_bounds__(256,3)
#define NREGS_4 __launch_bounds__(256,4)
#else
#define NREGS_0
#define NREGS_1
#define NREGS_2
#define NREGS_3
#define NREGS_4
#endif

/* ------------------------------------- */

//#define gpu_dmax2(k, l) LDG(Dcs[(k)*ncs+(l)])
__device__ inline float gpu_dmax2(const int k, const int l)
{
    return (2 * LDG(Dcs[k*ncs_mon+l]));
};

/* ------------------------------------- */
#ifdef SORT_IJ_SCHWARZ
#define CNV_CSP(a) sorted_csp[(a)]
#else
#define CNV_CSP(a) (a)
#endif

// nwks, iwk: wk=dev
__global__ void
NREGS_4
gpu_ifc4c_os_ssss( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  int tsize=0;
//  __shared__ double sV[nthb];
  volatile double *sV=&shared[tsize]; // sV
  tsize+=nthb;
//  double V[1];
//  double V;
  const int La=0, Lb=0, Lc=0, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, jcs, jat, jao;
//    int kcs, kat, kao, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    //double SSSS[1];
    double *SSSS = &shared[tsize+tidx]; // SSSS
    tsize+=nthb;

    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao    = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
	//IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
        sV[tidx]=ZERO;
        //V[0]=ZERO;
        //V=ZERO;
#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//#pragma omp for schedule(guided)
//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif
            int kcs, kat, kao, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao    = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_ssss_(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      SSSS );
	    double coe = (kao==lao? ONE : TWO );
	    int KL    = ((kao*kao+kao)>>1) + lao;
	    if ( fabs(SSSS[0]) > eps_eri ) {
//		V_frg[IJ] += coe * D_mon[KL] * SSSS[0];
		//V[0] += coe * LDG(D_mon[KL]) * SSSS[0];
		//V += coe * LDG(D_mon[KL]) * SSSS[0];
		sV[tidx] += coe * LDG(D_mon[KL]) * SSSS[0];
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
	int IJ     = ((iao*iao+iao)>>1) + jao;
        //sV[tidx]=V[0];
        //sV[tidx]=V;
        warpReduce(&sV[widx*WARP_SIZE], tidw);
//        if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
        __syncthreads();
        if (widx==0) { // assume nwrp <= WARP_SIZE
          if (tidx<nwrp) sV[tidx]=sV[tidx*WARP_SIZE];
          else sV[tidx]=ZERO;
          warpReduce(sV, tidx);
          if (tidx==0) V_frg[IJ]+=sV[0];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_ssss_


/** (ss,ps)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_4
gpu_ifc4c_os_ssps( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=0, Lb=0, Lc=1, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, jcs, jat, jao;
//    int kcs, kat, kao, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double SSPS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table1;
  for (int i=tidx; i<FMT_m_size[1]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[1];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  //double V;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao    = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
	//V=ZERO;
#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_psss_(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      SSPS );
            double V=ZERO;
#pragma unroll
	    for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++ ) {
		int KL = ((kao*kao+kao)>>1) + lao;
		if ( fabs(SSPS[k]) > eps_eri ) {
//		    V_frg[IJ] += TWO * D_mon[KL] * SSPS[k];
		    V += TWO * LDG(D_mon[KL]) * SSPS[k];
		    //sV[tidx] += TWO * D_mon[KL] * SSPS[k];
		}
	    }
            sV[tidx] += V;
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
	int IJ     = ((iao*iao+iao)>>1) + jao;
        //sV[tidx]=V[0];
        //sV[tidx]=V;
        warpReduce(&sV[widx*WARP_SIZE], tidw);
//        if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
        __syncthreads();
        if (widx==0) { // assume nwrp <= WARP_SIZE
          if (tidx<nwrp) sV[tidx]=sV[tidx*WARP_SIZE];
          else sV[tidx]=ZERO;
          warpReduce(sV, tidx);
          if (tidx==0) V_frg[IJ]+=sV[0];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_ssps_


/** (ss,pp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_3
gpu_ifc4c_os_sspp( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=0, Lb=0, Lc=1, Ld=1;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, jcs, jat, jao;
//    int kcs, kat, kao, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double SSPP[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table2;
  for (int i=tidx; i<FMT_m_size[2]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[2];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  //double V;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao    = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
	//V=ZERO;
#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_ppss_(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      SSPP );
            double V=ZERO;
#pragma unroll
	    for (int k=0, kao=kao0, ix=0; k<NNAO(Lc); k++, kao++ ) {
		int K2 = (kao*kao+kao)>>1;
#pragma unroll
		for (int l=0, lao=lao0; l<NNAO(Ld); l++, lao++, ix++ ) {
		    if ( lao>kao ) continue;
		    int KL = K2 + lao;
		    double coe = (kao==lao? ONE : TWO );
		    if ( fabs(SSPP[ix]) > eps_eri ) {
			//V_frg[IJ] += coe * D_mon[KL] * SSPP[ix];
			V += coe * LDG(D_mon[KL]) * SSPP[ix];
			//sV[tidx] += coe * D_mon[KL] * SSPP[ix];
		    }
		}
	    }
            sV[tidx] += V;
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
	int IJ     = ((iao*iao+iao)>>1) + jao;
        //sV[tidx]=V[0];
        //sV[tidx]=V;
        warpReduce(&sV[widx*WARP_SIZE], tidw);
//        if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
        __syncthreads();
        if (widx==0) { // assume nwrp <= WARP_SIZE
          if (tidx<nwrp) sV[tidx]=sV[tidx*WARP_SIZE];
          else sV[tidx]=ZERO;
          warpReduce(sV, tidx);
          if (tidx==0) V_frg[IJ]+=sV[0];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_sspp_


/** (ss,ds)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_3
gpu_ifc4c_os_ssds( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=0, Lb=0, Lc=2, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, jcs, jat, jao;
//    int kcs, kat, kao, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double SSDS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table2;
  for (int i=tidx; i<FMT_m_size[2]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[2];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  //double V;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao    = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
	//V=ZERO;
#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dsss(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      SSDS );
            double V=ZERO;
#pragma unroll
	    for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++ ) {
		int KL = ((kao*kao+kao)>>1) + lao;
		if ( fabs(SSDS[k]) > eps_eri ) {
//		    V_frg[IJ] += TWO * D_mon[KL] * SSDS[k];
		    V += TWO * LDG(D_mon[KL]) * SSDS[k];
		    //sV[tidx] += TWO * D_mon[KL] * SSDS[k];
		}
	    }
            sV[tidx] += V;
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
	int IJ     = ((iao*iao+iao)>>1) + jao;
        //sV[tidx]=V[0];
        //sV[tidx]=V;
        warpReduce(&sV[widx*WARP_SIZE], tidw);
//        if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
        __syncthreads();
        if (widx==0) { // assume nwrp <= WARP_SIZE
          if (tidx<nwrp) sV[tidx]=sV[tidx*WARP_SIZE];
          else sV[tidx]=ZERO;
          warpReduce(sV, tidx);
          if (tidx==0) V_frg[IJ]+=sV[0];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_ssds_


/** (ss,dp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_2
gpu_ifc4c_os_ssdp( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=0, Lb=0, Lc=2, Ld=1;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, jcs, jat, jao;
//    int kcs, kat, kao, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double SSDP[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table3;
  for (int i=tidx; i<FMT_m_size[3]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[3];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  //double V;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao    = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
	//V=ZERO;
#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0   = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dpss(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      SSDP );
            double V=ZERO;
#pragma unroll
	    for (int k=0, kao=kao0, ix=0; k<NNAO(Lc); k++, kao++ ) {
		int K2 = (kao*kao+kao)>>1;
#pragma unroll
		for (int l=0, lao=lao0; l<NNAO(Ld); l++, lao++, ix++ ) {
		    int KL = K2 + lao;
		    if ( fabs(SSDP[ix]) > eps_eri ) {
			//V_frg[IJ] += TWO * D_mon[KL] * SSDP[ix];
			V += TWO * LDG(D_mon[KL]) * SSDP[ix];
		    }
		}
	    }
            sV[tidx] += V;
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
	int IJ     = ((iao*iao+iao)>>1) + jao;
        //sV[tidx]=V[0];
        //sV[tidx]=V;
        warpReduce(&sV[widx*WARP_SIZE], tidw);
//        if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
        __syncthreads();
        if (widx==0) { // assume nwrp <= WARP_SIZE
          if (tidx<nwrp) sV[tidx]=sV[tidx*WARP_SIZE];
          else sV[tidx]=ZERO;
          warpReduce(sV, tidx);
          if (tidx==0) V_frg[IJ]+=sV[0];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_ssdp_


/** (ss,dd)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_3
gpu_ifc4c_os_ssdd( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=0, Lb=0, Lc=2, Ld=2;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, jcs, jat, jao;
//    int kcs, kat, kao, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double SSDD[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table4;
  for (int i=tidx; i<FMT_m_size[4]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[4];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  //double V;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao    = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
	//V=ZERO;
#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0   = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_ddss(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      SSDD );
            double V=ZERO;
#pragma unroll
	    for (int k=0, kao=kao0, ix=0; k<NNAO(Lc); k++, kao++ ) {
		int K2 = (kao*kao+kao)>>1;
#pragma unroll
		for (int l=0, lao=lao0; l<NNAO(Ld); l++, lao++, ix++ ) {
		    if ( lao>kao ) continue;
		    int KL = K2 + lao;
		    double coe = (kao==lao? ONE : TWO );
		    if ( fabs(SSDD[ix]) > eps_eri ) {
			//V_frg[IJ] += coe * D_mon[KL] * SSDD[ix];
			V += coe * LDG(D_mon[KL]) * SSDD[ix];
		    }
		}
	    }
            sV[tidx] += V;
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
	int IJ     = ((iao*iao+iao)>>1) + jao;
        //sV[tidx]=V[0];
        //sV[tidx]=V;
        warpReduce(&sV[widx*WARP_SIZE], tidw);
//        if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
        __syncthreads();
        if (widx==0) { // assume nwrp <= WARP_SIZE
          if (tidx<nwrp) sV[tidx]=sV[tidx*WARP_SIZE];
          else sV[tidx]=ZERO;
          warpReduce(sV, tidx);
          if (tidx==0) V_frg[IJ]+=sV[0];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_ssdd_


/** (ps,ss)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_4
gpu_ifc4c_os_psss( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=1, Lb=0, Lc=0, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double PSSS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table1;
  for (int i=tidx; i<FMT_m_size[1]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[1];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[3];
  volatile double *sVw=&shared[tsize];
  tsize+=nwrp*3;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	//sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<3; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao    = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_psss_(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      PSSS );
	    double coe = (kao==lao? ONE : TWO );
            int KL = ((kao*kao+kao)>>1) + lao;
#pragma unroll
	    for (int i=0; i<3; i++) {
		if ( fabs(PSSS[i]) > eps_eri ) {
		    V[i] += coe * LDG(D_mon[KL]) * PSSS[i];
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<3; i++) {
          sV[tidx]=V[i];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
            int iao = iao0 + i;
            //int IJ     = ((iao*iao+iao)>>1) + jao;
            //      if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
          if (tidw==0) sVw[i*nwrp+widx] = sV[tidx];
        }
        __syncthreads();
        for (int i=widx; i<3; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<3) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_psss_


/** (ps,ps)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_3
gpu_ifc4c_os_psps( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=1, Lb=0, Lc=1, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double PSPS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table2;
  for (int i=tidx; i<FMT_m_size[2]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[2];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  volatile double *sVw=&shared[tsize];
  tsize+=nwrp*Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_psps_(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      PSPS );
#pragma unroll
	    for (int i=0,ix=0; i<NNAO(La); i++) {
#pragma unroll
		for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++, ix++ ) {
		    int KL = ((kao*kao+kao)>>1) + lao;
		    if ( fabs(PSPS[ix]) > eps_eri ) {
			V[i] += TWO * LDG(D_mon[KL]) * PSPS[ix];
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          sV[tidx]=V[i];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
            int iao = iao0 + i;
            //int IJ     = ((iao*iao+iao)>>1) + jao;
            //      if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
          if (tidw==0) sVw[i*nwrp+widx] = sV[tidx];
        }
        __syncthreads();
        for (int i=widx; i<Nab; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<Nab) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_psps_


/** (ps,pp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_2
gpu_ifc4c_os_pspp( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=1, Lb=0, Lc=1, Ld=1;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double PSPP[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table3;
  for (int i=tidx; i<FMT_m_size[3]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[3];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  volatile double *sVw=&shared[tsize];
  tsize+=nwrp*Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0   = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_ppps_(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      PSPP );
#pragma unroll
	    for (int k=0, kao=kao0, ix=0; k<NNAO(Lc); k++, kao++ ) {
		int K2 = (kao*kao+kao)>>1;
#pragma unroll
		for (int l=0, lao=lao0; l<NNAO(Ld); l++, lao++ ) {
		    if ( lao>kao ) { ix+=NNAO(La); continue; }
		    int KL = K2 + lao;
		    double coe = (kao==lao? ONE : TWO );
#pragma unroll
		    for (int i=0; i<NNAO(La); i++, ix++ ) {
			if ( fabs(PSPP[ix]) > eps_eri ) {
			    //V_frg[IJ] += coe * D_mon[KL] * PSPP[ix];
			    V[i] += coe * LDG(D_mon[KL]) * PSPP[ix];
			}
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          sV[tidx]=V[i];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
            int iao = iao0 + i;
            //int IJ     = ((iao*iao+iao)>>1) + jao;
            //      if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
          if (tidw==0) sVw[i*nwrp+widx] = sV[tidx];
        }
        __syncthreads();
        for (int i=widx; i<Nab; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<Nab) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_pspp_


/** (ps,ds)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_2
gpu_ifc4c_os_psds( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=1, Lb=0, Lc=2, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double PSDS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table3;
  for (int i=tidx; i<FMT_m_size[3]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[3];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  volatile double *sVw=&shared[tsize];
  tsize+=nwrp*Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dsps(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      PSDS );
#pragma unroll
            for (int k=0, kao=kao0, ix=0; k<NNAO(Lc); k++, kao++ ) {
              int KL = ((kao*kao+kao)>>1) + lao;
#pragma unroll
              for (int i=0; i<NNAO(La); i++, ix++) {
		    if ( fabs(PSDS[ix]) > eps_eri ) {
			V[i] += TWO * LDG(D_mon[KL]) * PSDS[ix];
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          sV[tidx]=V[i];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
            int iao = iao0 + i;
            //int IJ     = ((iao*iao+iao)>>1) + jao;
            //      if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
          if (tidw==0) sVw[i*nwrp+widx] = sV[tidx];
        }
        __syncthreads();
        for (int i=widx; i<Nab; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<Nab) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_psds_


/** (ps,dp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_1
gpu_ifc4c_os_psdp( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=1, Lb=0, Lc=2, Ld=1;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double PSDP[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table4;
  for (int i=tidx; i<FMT_m_size[4]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[4];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  volatile double *sVw=&shared[tsize];
  tsize+=nwrp*Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0   = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dpps(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      PSDP );
#pragma unroll
	    for (int k=0, kao=kao0, ix=0; k<NNAO(Lc); k++, kao++ ) {
		int K2 = (kao*kao+kao)>>1;
#pragma unroll
		for (int l=0, lao=lao0; l<NNAO(Ld); l++, lao++ ) {
		    int KL = K2 + lao;
#pragma unroll
		    for (int i=0, iao=iao0; i<NNAO(La); i++, iao++, ix++ ) {
//			IJ = ((iao*iao+iao)>>1) + jao;
			if ( fabs(PSDP[ix]) > eps_eri ) {
//			    V_frg[IJ] += TWO * D_mon[KL] * PSDP[ix];
			V[i] += TWO * LDG(D_mon[KL]) * PSDP[ix];
			}
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          sV[tidx]=V[i];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
            int iao = iao0 + i;
            //int IJ     = ((iao*iao+iao)>>1) + jao;
            //      if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
          if (tidw==0) sVw[i*nwrp+widx] = sV[tidx];
        }
        __syncthreads();
        for (int i=widx; i<Nab; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<Nab) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_psdp_


#if 0
/** (ps,dd)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_psdd(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, k, l, K2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], CA[3];
    double val_ab, val_cd, coe;
    double PSDD[3*6*6];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao0   = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_ddps( &La, &Lb, &Lc, &Ld,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      PSDD );
	    for ( k=0, kao=kao0, ix=0; k<6; k++, kao++ ) {
		K2 = (kao*kao+kao)>>1;
		for ( l=0, lao=lao0; l<6; l++, lao++ ) {
		    if ( lao>kao ) { ix+=3; continue; }
		    KL = K2 + lao;
		    coe = (kao==lao? ONE : TWO );
		    for ( i=0, iao=iao0; i<3; i++, iao++, ix++ ) {
			IJ = ((iao*iao+iao)>>1) + jao;
			if ( fabs(PSDD[ix]) > eps_eri ) {
			    V_frg[IJ] += coe * D_mon[KL] * PSDD[ix];
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_psdd_
#endif // if 0


/** (pp,ss)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_3
gpu_ifc4c_os_ppss( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=1, Lb=1, Lc=0, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double PPSS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table2;
  for (int i=tidx; i<FMT_m_size[2]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[2];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao    = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_ppss_(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      PPSS );
	    double coe = (kao==lao? ONE : TWO );
            int KL = ((kao*kao+kao)>>1) + lao;
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
//		int I2 = (iao*iao+iao)>>1;
#pragma unroll
		for (int j=0, jao=jao0; j<NNAO(Lb); j++, jao++, ix++ ) {
                  //int ij=i*3+j;
		    if ( jao>iao ) continue;
		    //IJ = I2 + jao;
		    if ( fabs(PPSS[ix]) > eps_eri ) {
		//	V_frg[IJ] += coe * D_mon[KL] * PPSS[ix];
                      V[ix] += coe * LDG(D_mon[KL]) * PPSS[ix];
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
	for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i/NNAO(Lb);
          int jao = jao0 + i%NNAO(Lb);
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_ppss_


/** (pp,ps)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_2
gpu_ifc4c_os_ppps( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=1, Lb=1, Lc=1, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double PPPS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table3;
  for (int i=tidx; i<FMT_m_size[3]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[3];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_ppps_(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      PPPS );
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
//		int I2 = (iao*iao+iao)>>1;
#pragma unroll
		for (int j=0, jao=jao0; j<NNAO(Lb); j++, jao++ ) {
                  int ij=i*3+j;
                  double Vt=ZERO;
		    if ( jao>iao ) { ix+=NNAO(Lc); continue; }
#pragma unroll
		    for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++, ix++ ) {
			int KL = ((kao*kao+kao)>>1) + lao;
			if ( fabs(PPPS[ix]) > eps_eri ) {
//			    V_frg[IJ] += TWO * D_mon[KL] * PPPS[ix];
			    //Vt += TWO * D_mon[KL] * PPPS[ix];
			    V[ij] += TWO * LDG(D_mon[KL]) * PPPS[ix];
			}
		    }
		  //int IJ = I2 + jao;
                  //atomicAdd(&V_frg[IJ], Vt);
                  //V[ij] += Vt;
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs

#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
	for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i/NNAO(Lb);
          int jao = jao0 + i%NNAO(Lb);
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_ppps_


/** (pp,pp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_1
gpu_ifc4c_os_pppp( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=1, Lb=1, Lc=1, Ld=1;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double PPPP[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table4;
  for (int i=tidx; i<FMT_m_size[4]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[4];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_pppp_(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      PPPP );
//#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
		//int I2 = (iao*iao+iao)>>1;
//#pragma unroll
		for (int j=0, jao=jao0; j<NNAO(Lb); j++, jao++ ) {
		    if ( jao>iao ) { ix+=Ncd; continue; }
		    //int IJ = I2 + jao;
                    double Vt=ZERO;
#pragma unroll
		    for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++ ) {
			int K2 = (kao*kao+kao)>>1;
#pragma unroll
			for (int l=0, lao=lao0; l<NNAO(Ld); l++, lao++, ix++ ) {
			    if ( lao>kao ) continue;
			    double coe = (kao==lao? ONE : TWO );
			    int KL = K2 + lao;
			    if ( fabs(PPPP[ix]) > eps_eri ) {
				//V_frg[IJ] += coe * D_mon[KL] * PPPP[ix];
				Vt += coe * LDG(D_mon[KL]) * PPPP[ix];
			    }
			}
		    }
                    int ij = i*NNAO(Lb) + j;
                    V[ij] += Vt;
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
	for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i/NNAO(Lb);
          int jao = jao0 + i%NNAO(Lb);
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_pppp_


/** (pp,ds)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_1
gpu_ifc4c_os_ppds( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=1, Lb=1, Lc=2, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double PPDS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table4;
  for (int i=tidx; i<FMT_m_size[4]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[4];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dspp(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      PPDS );
#pragma unroll
	    for (int k=0, kao=kao0, ix=0; k<NNAO(Lc); k++, kao++ ) {
		int KL = ((kao*kao+kao)>>1) + lao;
#pragma unroll
		for (int i=0, iao=iao0; i<NNAO(La); i++, iao++ ) {
		    //int I2 = (iao*iao+iao)>>1;
#pragma unroll
		    for (int j=0, jao=jao0; j<NNAO(Lb); j++, jao++, ix++ ) {
			if ( jao>iao ) continue;
			//int IJ = I2 + jao;
			if ( fabs(PPDS[ix]) > eps_eri ) {
			    //V_frg[IJ] += TWO * D_mon[KL] * PPDS[ix];
                            int ij = i*NNAO(Lb) + j;
			    V[ij] += TWO * LDG(D_mon[KL]) * PPDS[ix];
			}
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
	for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i/NNAO(Lb);
          int jao = jao0 + i%NNAO(Lb);
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_ppds_

#if 0

/** (pp,dp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_ppdp(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, j, k, l, I2, K2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], CA[3];
    double val_ab, val_cd;
    double PPDP[3*3*6*3];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao0   = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_dppp( &La, &Lb, &Lc, &Ld,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      PPDP );
	    for ( k=0, kao=kao0, ix=0; k<6; k++, kao++ ) {
		K2 = (kao*kao+kao)>>1;
		for ( l=0, lao=lao0; l<3; l++, lao++ ) {
		    KL = K2 + lao;
		    for ( i=0, iao=iao0; i<3; i++, iao++ ) {
			I2 = (iao*iao+iao)>>1;
			for ( j=0, jao=jao0; j<3; j++, jao++, ix++ ) {
			    if ( jao>iao ) continue;
			    IJ = I2 + jao;
			    if ( fabs(PPDP[ix]) > eps_eri ) {
				V_frg[IJ] += TWO * D_mon[KL] * PPDP[ix];
			    }
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_ppdp_


/** (pp,dd)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_ppdd(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, j, k, l, I2, K2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], CA[3];
    double val_ab, val_cd, coe;
    double PPDD[3*3*6*6];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao0   = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_ddpp( &La, &Lb, &Lc, &Ld,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      PPDD );
	    for ( k=0, kao=kao0, ix=0; k<6; k++, kao++ ) {
		K2 = (kao*kao+kao)>>1;
		for ( l=0, lao=lao0; l<6; l++, lao++ ) {
		    if ( lao>kao ) { ix+=3*3; continue; }
		    KL = K2 + lao;
		    coe = (kao==lao? ONE : TWO );
		    for ( i=0, iao=iao0; i<3; i++, iao++ ) {
			I2 = (iao*iao+iao)>>1;
			for ( j=0, jao=jao0; j<3; j++, jao++, ix++ ) {
			    if ( jao>iao ) continue;
			    IJ = I2 + jao;
			    if ( fabs(PPDD[ix]) > eps_eri ) {
				V_frg[IJ] += coe * D_mon[KL] * PPDD[ix];
			    }
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_ppdd_
#endif /* if 0 */


/** (ds,ss)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_3
gpu_ifc4c_os_dsss( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=0, Lc=0, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DSSS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table2;
  for (int i=tidx; i<FMT_m_size[2]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[2];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  volatile double *sVw=&shared[tsize];
  tsize+=nwrp*Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao    = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_dsss_(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DSSS );
	    double coe = (kao==lao? ONE : TWO );
            int KL = ((kao*kao+kao)>>1) + lao;
#pragma unroll
	    for (int i=0; i<Nab; i++) {
		if ( fabs(DSSS[i]) > eps_eri ) {
		    V[i] += coe * D_mon[KL] * DSSS[i];
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          sV[tidx]=V[i];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
            int iao = iao0 + i;
            //int IJ     = ((iao*iao+iao)>>1) + jao;
            //      if (tidw==0) atomicAdd(&V_frg[IJ], sV[tidx]);
          if (tidw==0) sVw[i*nwrp+widx] = sV[tidx];
        }
        __syncthreads();
        for (int i=widx; i<Nab; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<Nab) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dsss_


/** (ds,ps)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_2
gpu_ifc4c_os_dsps( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=0, Lc=1, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DSPS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table3;
  for (int i=tidx; i<FMT_m_size[3]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[3];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
//  volatile double *sVw=&shared[tsize];
//  tsize+=nwrp*Nab;
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dsps(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DSPS );
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
//		IJ = ((iao*iao+iao)>>1) + jao;
#pragma unroll
		for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++, ix++ ) {
		    if ( fabs(DSPS[ix]) > eps_eri ) {
                      int KL = ((kao*kao+kao)>>1) + lao;
//			V_frg[IJ] += TWO * D_mon[KL] * DSPS[ix];
			V[i] += TWO * LDG(D_mon[KL]) * DSPS[ix];
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
#if 0
        for (int i=widx; i<Nab; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<Nab) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#endif
        for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i;
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dsps_


/** (ds,pp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_1
gpu_ifc4c_os_dspp( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=0, Lc=1, Ld=1;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DSPP[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table4;
  for (int i=tidx; i<FMT_m_size[4]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[4];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
//  volatile double *sVw=&shared[tsize];
//  tsize+=nwrp*Nab;
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0   = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dspp(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DSPP );
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
#pragma unroll
		for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++ ) {
		    int K2 = (kao*kao+kao)>>1;
#pragma unroll
		    for (int l=0, lao=lao0; l<NNAO(Ld); l++, lao++, ix++ ) {
			if ( lao>kao ) continue;
			double coe = (kao==lao? ONE : TWO );
			int KL = K2 + lao;
			if ( fabs(DSPP[ix]) > eps_eri ) {
			    //V_frg[IJ] += coe * D_mon[KL] * DSPP[ix];
                            V[i] += coe * LDG(D_mon[KL]) * DSPP[ix];
			}
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
#if 0
        for (int i=widx; i<Nab; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<Nab) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#endif
        for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i;
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dspp_


/** (ds,ds)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_1
gpu_ifc4c_os_dsds( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=0, Lc=2, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DSDS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table4;
  for (int i=tidx; i<FMT_m_size[4]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[4];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
//  volatile double *sVw=&shared[tsize];
//  tsize+=nwrp*Nab;
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dsds(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DSDS );
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
//		IJ = ((iao*iao+iao)>>1) + jao;
#pragma unroll
		for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++, ix++ ) {
		    if ( fabs(DSDS[ix]) > eps_eri ) {
                      int KL = ((kao*kao+kao)>>1) + lao;
//			V_frg[IJ] += TWO * D_mon[KL] * DSDS[ix];
			V[i] += TWO * LDG(D_mon[KL]) * DSDS[ix];
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
#if 0
        for (int i=widx; i<Nab; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<Nab) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#endif
        for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i;
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dsds_


/** (ds,dp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_1
gpu_ifc4c_os_dsdp( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=0, Lc=2, Ld=1;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao, iao0, jcs, jat, jao;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DSDP[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table5;
  for (int i=tidx; i<FMT_m_size[5]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[5];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
//  volatile double *sVw=&shared[tsize];
//  tsize+=nwrp*Nab;
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
//	IJ     = ((iao*iao+iao)>>1) + jao;
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
	sV[tidx]=ZERO;
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0   = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], CA[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dpds(
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      DSDP );
#pragma unroll
	    for (int k=0, kao=kao0, ix=0; k<NNAO(Lc); k++, kao++ ) {
		int K2 = (kao*kao+kao)>>1;
#pragma unroll
		for (int l=0, lao=lao0; l<NNAO(Ld); l++, lao++ ) {
		    int KL = K2 + lao;
#pragma unroll
		    for (int i=0, iao=iao0; i<NNAO(La); i++, iao++, ix++ ) {
			//IJ = ((iao*iao+iao)>>1) + jao;
			if ( fabs(DSDP[ix]) > eps_eri ) {
			    //V_frg[IJ] += TWO * D_mon[KL] * DSDP[ix];
			    V[i] += TWO * LDG(D_mon[KL]) * DSDP[ix];
			}
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
#if 0
        for (int i=widx; i<Nab; i+=nwrp) {
          if (tidw<nwrp) sV[tidx]=sVw[i*nwrp+tidw];
          else sV[tidx]=ZERO;
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          /*
          if (tidw==0) {
            int iao = iao0 + i;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx];
          }
          */
        }
        __syncthreads();
        if (tidx<Nab) {
            int iao = iao0 + tidx;
            int IJ     = ((iao*iao+iao)>>1) + jao;
            V_frg[IJ]+=sV[tidx*WARP_SIZE];
        }
#endif
        for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i;
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dsdp_

#if 0

/** (ds,dd)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_dsdd(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, k, l, K2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], CA[3];
    double val_ab, val_cd, coe;
    double DSDD[6*6*6];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao    = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao0   = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_ddds( &La, &Lb, &Lc, &Ld,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      DSDD );
	    for ( k=0, kao=kao0, ix=0; k<6; k++, kao++ ) {
		K2 = (kao*kao+kao)>>1;
		for ( l=0, lao=lao0; l<6; l++, lao++ ) {
		    if ( lao>kao ) { ix+=6; continue; }
		    KL = K2 + lao;
		    coe = (kao==lao? ONE : TWO );
		    for ( i=0, iao=iao0; i<6; i++, iao++, ix++ ) {
			IJ = ((iao*iao+iao)>>1) + jao;
			if ( fabs(DSDD[ix]) > eps_eri ) {
			    V_frg[IJ] += coe * D_mon[KL] * DSDD[ix];
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_dsdd_
#endif // if 0


/** (dp,ss)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_2
gpu_ifc4c_os_dpss( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=1, Lc=0, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DPSS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table3;
  for (int i=tidx; i<FMT_m_size[3]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[3];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao    = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dpss(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DPSS );
	    double coe = (kao==lao? ONE : TWO );
            int KL = ((kao*kao+kao)>>1) + lao;
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
#pragma unroll
		for (int j=0, jao=jao0; j<NNAO(Lb); j++, jao++, ix++ ) {
		    if ( fabs(DPSS[ix]) > eps_eri ) {
		//	V_frg[IJ] += coe * D_mon[KL] * DPSS[ix];
                      V[ix] += coe * LDG(D_mon[KL]) * DPSS[ix];
		    }
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
	for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i/NNAO(Lb);
          int jao = jao0 + i%NNAO(Lb);
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dpss_


/** (dp,ps)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_1
gpu_ifc4c_os_dpps( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=1, Lc=1, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DPPS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table4;
  for (int i=tidx; i<FMT_m_size[4]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[4];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dpps(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DPPS );
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
//		int I2 = (iao*iao+iao)>>1;
#pragma unroll
		for (int j=0, jao=jao0; j<NNAO(Lb); j++, jao++ ) {
                    double Vt=ZERO;
#pragma unroll
		    for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++, ix++ ) {
			int KL = ((kao*kao+kao)>>1) + lao;
			if ( fabs(DPPS[ix]) > eps_eri ) {
//			    V_frg[IJ] += TWO * D_mon[KL] * DPPS[ix];
			    Vt += TWO * LDG(D_mon[KL]) * DPPS[ix];
			    //V[ij] += TWO * D_mon[KL] * DPPS[ix];
			}
		    }
		  //int IJ = I2 + jao;
                  //atomicAdd(&V_frg[IJ], Vt);
                  int ij=i*NNAO(Lb)+j;
                  V[ij] += Vt;
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
	for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i/NNAO(Lb);
          int jao = jao0 + i%NNAO(Lb);
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dpps_


/** (dp,pp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_1
gpu_ifc4c_os_dppp( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=1, Lc=1, Ld=1;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DPPP[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table5;
  for (int i=tidx; i<FMT_m_size[5]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[5];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0   = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dppp(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DPPP );
//#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
		//int I2 = (iao*iao+iao)>>1;
//#pragma unroll
		for (int j=0, jao=jao0; j<NNAO(Lb); j++, jao++ ) {
		    //int IJ = I2 + jao;
                    double Vt=ZERO;
#pragma unroll
		    for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++ ) {
			int K2 = (kao*kao+kao)>>1;
#pragma unroll
			for (int l=0, lao=lao0; l<NNAO(Ld); l++, lao++, ix++ ) {
			    if ( lao>kao ) continue;
			    double coe = (kao==lao? ONE : TWO );
			    int KL = K2 + lao;
			    if ( fabs(DPPP[ix]) > eps_eri ) {
				//V_frg[IJ] += coe * D_mon[KL] * DPPP[ix];
				Vt += coe * LDG(D_mon[KL]) * DPPP[ix];
			    }
			}
		    }
                    int ij=i*NNAO(Lb)+j;
                    V[ij] += Vt;
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
	for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i/NNAO(Lb);
          int jao = jao0 + i%NNAO(Lb);
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dppp_


/** (dp,ds)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_0
gpu_ifc4c_os_dpds( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=1, Lc=2, Ld=0;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DPDS[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table5;
  for (int i=tidx; i<FMT_m_size[5]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[5];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao    = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dpds(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DPDS );
#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
//		int I2 = (iao*iao+iao)>>1;
#pragma unroll
		for (int j=0, jao=jao0; j<NNAO(Lb); j++, jao++ ) {
		    //int IJ = I2 + jao;
                    double Vt=ZERO;
#pragma unroll
		    for (int k=0, kao=kao0; k<NNAO(Lc); k++, kao++, ix++ ) {
			int KL = ((kao*kao+kao)>>1) + lao;
			if ( fabs(DPDS[ix]) > eps_eri ) {
			    //V_frg[IJ] += TWO * D_mon[KL] * DPDS[ix];
			    Vt += TWO * LDG(D_mon[KL]) * DPDS[ix];
			}
		    }
		  //int IJ = I2 + jao;
                  //atomicAdd(&V_frg[IJ], Vt);
                  int ij=i*NNAO(Lb)+j;
                  V[ij] += Vt;
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
	for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i/NNAO(Lb);
          int jao = jao0 + i%NNAO(Lb);
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dpds_


/** (dp,dp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
__global__ void
NREGS_0
gpu_ifc4c_os_dpdp( const int nwks, const int iwk,
    const float eps_eri, const float eps_ps4, const float eps_sch )
{
  int tidx = threadIdx.x + threadIdx.y * blockDim.x;
  int nthb = blockDim.x * blockDim.y;
  int bidx = blockIdx.x;
  int nblk = gridDim.x;
  int widx = threadIdx.y;
  int tidw = threadIdx.x;
  int nwrp = blockDim.y;
  int nwkblk = nwks * nblk;
//  int iwkblk = iwk * nblk + bidx;
  int iwkblk = bidx * nwks + iwk;
  const int La=2, Lb=1, Lc=2, Ld=1;
//    int Lab, Lcd, IJ, KL;
    int Lab, Lcd;
    int ijcs0, ijcs1;
    int klcs0, klcs1;
//    int ijps0, nijps, klps0, nklps;
    int ijps0, nijps;
    int ics, iat, iao0, jcs, jat, jao0;
//    int kcs, kat, kao, kao0, lcs, lat, lao;
//    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double A[3], B[3], BA[3];
    //double val_ab, val_cd, coe;
    const int Nab=NNAO(La)*NNAO(Lb);
    const int Ncd=NNAO(Lc)*NNAO(Ld);
    double DPDP[Nab*Ncd];

  int tsize=0;
#ifdef CUDA_FMT_M_SM
  double *tbl = FMT_m_table6;
  for (int i=tidx; i<FMT_m_size[6]; i+=nthb) shared[tsize+i] = tbl[i];
  tsize += FMT_m_size[6];
  __syncthreads();
#endif
  volatile double *sV=&shared[tsize];
  tsize+=nthb;
  double V[Nab];
  double *sVw=&shared[tsize];
  tsize+=Nab;
    //int ixx, ncsp, dx, res, pos;
    int *sklcs = sklcs_b + bidx * max_num_klcs;
    __shared__ int nklcs;
#ifdef DLB_KL
    volatile int *klcsw = (int *)&shared[tsize];
    tsize += nwrp;
    __shared__ int cklcs;
#endif

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
//    ijcs0 = leading_cs_pair_frg[Lab] + iwkblk;
    ijcs0 = leading_cs_pair_frg[Lab];
    ijcs1 = leading_cs_pair_frg[Lab+1];
    //ixx   = nwkblk;
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
#ifdef GPU_DLB
    int Labcd=Lab*6+Lcd;
    int *p_ijcounter = &ijcounter[Labcd];
    __shared__ int ijcsw;
    if (tidx==0) ijcsw = ijcs0+iwk + nwks*bidx;
#ifdef DLB_KL
    if (tidx==0) cklcs=nthb;
#endif
    for (int i=tidx; i<Nab; i+=nthb ) sVw[i] = ZERO;
    __syncthreads();
//    int ijcs = ijcsw;
    while(ijcsw<ijcs1) {
#else // !GPU_DLB
//    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
    for (int ijcsw=ijcs0+iwkblk; ijcsw<ijcs1; ijcsw+=nwkblk ) {
#endif // GPU_DLB
      int ijcs = CNV_CSP(ijcsw);
	float val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
#pragma unroll
	for (int i=0; i<3; i++ ) BA[i] = B[i] - A[i];
#pragma unroll
	for (int i=0; i<Nab; i++ ) V[i] = ZERO;

#ifndef USE_INSTANT_SCHWARZ
        {
          if(tidx==0) nklcs=0;
          __syncthreads();
	  for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
            int iklcs = atomicAdd(&nklcs, 1);
            sklcs[iklcs] = klcs;
          }
          __threadfence_block();
          __syncthreads();
        }
#endif /* USE_INSTANT_SCHWARZ */

//	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
#ifdef USE_INSTANT_SCHWARZ
	for (int klcs=klcs0+tidx; klcs<klcs1; klcs+=nthb ) {
	    float val = val_ab * csp_schwarz_mon[klcs];
	    if ( val < eps_ps4 ) continue;
            val *= gpu_dmax2(csp_ics_mon[klcs], csp_jcs_mon[klcs]);
	    if ( val < eps_sch ) continue;
#else
#ifndef DLB_KL
	for (int iklcs=tidx; iklcs<nklcs; iklcs+=nthb ) {
#else
        int iklcs=tidx;
        while (iklcs<nklcs) {
#endif
            int klcs = sklcs[iklcs];
#endif /* USE_INSTANT_SCHWARZ */
            int kcs, kat, kao0, lcs, lat, lao0;
            int klps0, nklps;
	    kcs    = LDG(csp_ics_mon[klcs]);
	    lcs    = LDG(csp_jcs_mon[klcs]);
	    klps0  = LDG(csp_leading_ps_pair_mon[klcs]);
	    nklps  = LDG(csp_leading_ps_pair_mon[klcs+1])-klps0;
	    kat    = LDG(shel_atm_mon[kcs]);
	    lat    = LDG(shel_atm_mon[lcs]);
	    kao0   = LDG(shel_ini_mon[kcs]);
	    lao0   = LDG(shel_ini_mon[lcs]);
            double C[3], D[3], DC[3], AC[3];
	    C[0]=LDG(atom_x_mon[kat]); C[1]=LDG(atom_y_mon[kat]); C[2]=LDG(atom_z_mon[kat]);
	    D[0]=LDG(atom_x_mon[lat]); D[1]=LDG(atom_y_mon[lat]); D[2]=LDG(atom_z_mon[lat]);
#pragma unroll
	    for (int i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    gpu_twoint_core_os_dpdp(
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DPDP );
//#pragma unroll
	    for (int i=0, iao=iao0, ix=0; i<NNAO(La); i++, iao++ ) {
		//int I2 = (iao*iao+iao)>>1;
//#pragma unroll
		for (int j=0, jao=jao0; j<NNAO(Lb); j++, jao++ ) {
		    //int IJ = I2 + jao;
                    double Vt=ZERO;
#pragma unroll
		    for (int k=0, kao=kao0; k<6; k++, kao++ ) {
			int K2 = (kao*kao+kao)>>1;
#pragma unroll
			for (int l=0, lao=lao0; l<3; l++, lao++, ix++ ) {
			    int KL = K2 + lao;
			    if ( fabs(DPDP[ix]) > eps_eri ) {
				//V_frg[IJ] += TWO * D_mon[KL] * DPDP[ix];
				Vt += TWO * LDG(D_mon[KL]) * DPDP[ix];
			    }
			}
		    }
		    int ij = i*NNAO(Lb)+j;
                    V[ij] += Vt;
		}
	    }
#ifdef DLB_KL
            if (tidw==0) klcsw[widx] = atomicAdd(&cklcs, warpSize);
            iklcs = klcsw[widx]+tidw;
#endif
	}	// klcs
#pragma unroll
        for (int i=0; i<Nab; i++) {
          int it = (i+widx)%Nab;
          sV[tidx]=V[it];
          warpReduce(&sV[widx*WARP_SIZE], tidw);
          if (tidw==0) atomicAdd(&sVw[it], sV[tidx]);
        }
        __syncthreads();
	for (int i=tidx; i<Nab; i+=nthb ) {
          int iao = iao0 + i/NNAO(Lb);
          int jao = jao0 + i%NNAO(Lb);
          int IJ     = ((iao*iao+iao)>>1) + jao;
          V_frg[IJ]+=sVw[i];
          sVw[i] = ZERO;
        }
#ifdef GPU_DLB
        if (tidx==0) {
          ijcsw = ijcs0+iwk + atomicAdd(p_ijcounter, 1)*nwks;
        }
#ifdef DLB_KL
        if (tidx==0) cklcs=nthb;
#endif
//        ijcs = CNV_CSP(ijcsw);
#endif // GPU_DLB
        __syncthreads();
    }		// ijcs
    return;
}	// end of ofmo_ifc4c_dpdp_


#if 0
/** (dp,dd)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_dpdd(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, j, k, l, I2, K2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], CA[3];
    double val_ab, val_cd, coe;
    double DPDD[6*3*6*6];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao0   = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		CA[i] = C[i] - A[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_dddp( &La, &Lb, &Lc, &Ld,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,   CA,      DPDD );
	    for ( k=0, kao=kao0, ix=0; k<6; k++, kao++ ) {
		K2 = (kao*kao+kao)>>1;
		for ( l=0, lao=lao0; l<6; l++, lao++ ) {
		    if ( lao>kao ) { ix+=6*3; continue; }
		    KL = K2 + lao;
		    coe = (kao==lao? ONE : TWO );
		    for ( i=0, iao=iao0; i<6; i++, iao++ ) {
			I2 = (iao*iao+iao)>>1;
			for ( j=0, jao=jao0; j<3; j++, jao++, ix++ ) {
			    IJ = I2 + jao;
			    if ( fabs(DPDD[ix]) > eps_eri ) {
				V_frg[IJ] += coe * D_mon[KL] * DPDD[ix];
			    }
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_dpdd_


/** (dd,ss)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_ddss(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, j, I2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, lcs, lat, lao;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DDSS[6*6];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao    = shel_ini_mon[kcs];
	    lao    = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    KL    = ((kao*kao+kao)>>1) + lao;
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_ddss( &La, &Lb, &Lc, &Ld,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DDSS );
	    coe = (kao==lao? ONE : TWO );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		I2 = (iao*iao+iao)>>1;
		for ( j=0, jao=jao0; j<6; j++, jao++, ix++ ) {
		    if ( jao>iao ) continue;
		    IJ = I2 + jao;
		    if ( fabs(DDSS[ix]) > eps_eri ) {
			V_frg[IJ] += coe * D_mon[KL] * DDSS[ix];
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_ddss_


/** (dd,ps)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_ddps(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, j, k, I2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd;
    double DDPS[6*6*3];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao    = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_ddps( &La, &Lb, &Lc, &Ld,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DDPS );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		I2 = (iao*iao+iao)>>1;
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix+=3; continue; }
		    IJ = I2 + jao;
		    for ( k=0, kao=kao0; k<3; k++, kao++, ix++ ) {
			KL = ((kao*kao+kao)>>1) + lao;
			if ( fabs(DDPS[ix]) > eps_eri ) {
			    V_frg[IJ] += TWO * D_mon[KL] * DDPS[ix];
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_ddps_


/** (dd,pp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_ddpp(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, j, k, l, I2, K2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DDPP[6*6*3*3];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao0   = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_ddpp( &La, &Lb, &Lc, &Ld,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DDPP );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		I2 = (iao*iao+iao)>>1;
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix+=3*3; continue; }
		    IJ = I2 + jao;
		    for ( k=0, kao=kao0; k<3; k++, kao++ ) {
			K2 = (kao*kao+kao)>>1;
			for ( l=0, lao=lao0; l<3; l++, lao++, ix++ ) {
			    if ( lao>kao ) continue;
			    coe = (kao==lao? ONE : TWO );
			    KL = K2 + lao;
			    if ( fabs(DDPP[ix]) > eps_eri ) {
				V_frg[IJ] += coe * D_mon[KL] * DDPP[ix];
			    }
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_ddpp_


/** (dd,ds)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_ddds(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, j, k, I2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd;
    double DDDS[6*6*6];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao    = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_ddds( &La, &Lb, &Lc, &Ld,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DDDS );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		I2 = (iao*iao+iao)>>1;
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix+=6; continue; }
		    IJ = I2 + jao;
		    for ( k=0, kao=kao0; k<6; k++, kao++, ix++ ) {
			KL = ((kao*kao+kao)>>1) + lao;
			if ( fabs(DDDS[ix]) > eps_eri ) {
			    V_frg[IJ] += TWO * D_mon[KL] * DDDS[ix];
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_ddds_


/** (dd,dp)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_dddp(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, j, k, l, I2, K2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd;
    double DDDP[6*6*6*3];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao0   = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_dddp( &La, &Lb, &Lc, &Ld,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DDDP );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		I2 = (iao*iao+iao)>>1;
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix+=6*3; continue; }
		    IJ = I2 + jao;
		    for ( k=0, kao=kao0; k<6; k++, kao++ ) {
			K2 = (kao*kao+kao)>>1;
			for ( l=0, lao=lao0; l<3; l++, lao++, ix++ ) {
			    KL = K2 + lao;
			    if ( fabs(DDDP[ix]) > eps_eri ) {
				V_frg[IJ] += TWO * D_mon[KL] * DDDP[ix];
			    }
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_dddp_


/** (dd,dd)タイプの４中心クーロンポテンシャル項をまとめて計算する
 * @ingroup integ-ifc4c
 * */
int ofmo_ifc4c_os_dddd(
	// parallelization
	const int *pnworkers, const int *pworkerid,
	// integral type data
	const int *pLa, const int *pLb, const int *pLc, const int *pLd,
	// basis and cutoff table data for fragment
	const int shel_atm_frg[], const int shel_ini_frg[],
	const double atom_x_frg[], const double atom_y_frg[],
	const double atom_z_frg[], const int leading_cs_pair_frg[],
	const double csp_schwarz_frg[],
	const int csp_ics_frg[], const int csp_jcs_frg[],
	const int csp_leading_ps_pair_frg[],
	const double psp_zeta_frg[], const double psp_dkps_frg[],
	const double psp_xiza_frg[],
	// basis and cutoff table data for monomer
	const int shel_atm_mon[], const int shel_ini_mon[],
	const double atom_x_mon[], const double atom_y_mon[],
	const double atom_z_mon[],
	const int leading_cs_pair_mon[],
	const double csp_schwarz_mon[],
	const int csp_ics_mon[], const int csp_jcs_mon[],
	const int csp_leading_ps_pair_mon[],
	const double psp_zeta_mon[], const double psp_dkps_mon[],
	const double psp_xiza_mon[],
	// density matrix of monomer
	const double D_mon[],
	// (output) Coulomb potential
	double V_frg[] ) {
	    int nworkers=*pnworkers, workerid=*pworkerid;
	    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int Lab, Lcd, IJ, KL, i, j, k, l, I2, K2, ix;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe;
    double DDDD[6*6*6*6];
    //
    int ixx, ncsp, dx, res, pos;
    float eps_eri = ofmo_twoint_eps_eri(0);
    float eps_ps4 = ofmo_twoint_eps_ps4(0);
    float eps_sch = ofmo_twoint_eps_sch(0);

    Lab = La * (La+1)/2 + Lb;
    Lcd = Lc * (Lc+1)/2 + Ld;
    if ( nworkers < 0 ) {
	pos   = workerid;
	ncsp  = leading_cs_pair_frg[Lab+1] - leading_cs_pair_frg[Lab];
	dx    = (ncsp>>7);	// >>5 means /32
	res   = (ncsp & 0x007f);// residues in division by 32
	ijcs0 = leading_cs_pair_frg[Lab]
	    + (pos<res ? pos*(dx+1) : pos*dx+res );
	ijcs1 = ijcs0 + ( pos<res ? dx+1 : dx );
	ixx   = 1;
    } else {
	ijcs0 = leading_cs_pair_frg[Lab] + workerid;
	ijcs1 = leading_cs_pair_frg[Lab+1];
	ixx   = nworkers;
    }
    klcs0 = leading_cs_pair_mon[Lcd];
    klcs1 = leading_cs_pair_mon[Lcd+1];
    for ( ijcs=ijcs0; ijcs<ijcs1; ijcs+=ixx ) {
	val_ab = csp_schwarz_frg[ijcs];
	ics    = csp_ics_frg[ijcs];
	jcs    = csp_jcs_frg[ijcs];
	ijps0  = csp_leading_ps_pair_frg[ijcs];
	nijps  = csp_leading_ps_pair_frg[ijcs+1]-ijps0;
	iat    = shel_atm_frg[ics];
	jat    = shel_atm_frg[jcs];
	iao0   = shel_ini_frg[ics];
	jao0   = shel_ini_frg[jcs];
	A[0]=atom_x_frg[iat]; A[1]=atom_y_frg[iat]; A[2]=atom_z_frg[iat];
	B[0]=atom_x_frg[jat]; B[1]=atom_y_frg[jat]; B[2]=atom_z_frg[jat];
	for ( i=0; i<3; i++ ) BA[i] = B[i] - A[i];

//#pragma omp for schedule(guided)
	for ( klcs=klcs0; klcs<klcs1; klcs++ ) {
	    val_cd = csp_schwarz_mon[klcs];
	    if ( val_ab*val_cd < eps_ps4 ) continue;
	    kcs    = csp_ics_mon[klcs];
	    lcs    = csp_jcs_mon[klcs];
	    if ( val_ab*val_cd*ofmo_twoint_dmax2(kcs,lcs) < eps_sch ) continue;
	    klps0  = csp_leading_ps_pair_mon[klcs];
	    nklps  = csp_leading_ps_pair_mon[klcs+1]-klps0;
	    kat    = shel_atm_mon[kcs];
	    lat    = shel_atm_mon[lcs];
	    kao0   = shel_ini_mon[kcs];
	    lao0   = shel_ini_mon[lcs];
	    C[0]=atom_x_mon[kat]; C[1]=atom_y_mon[kat]; C[2]=atom_z_mon[kat];
	    D[0]=atom_x_mon[lat]; D[1]=atom_y_mon[lat]; D[2]=atom_z_mon[lat];
	    for ( i=0; i<3; i++ ) {
		AC[i] = A[i] - C[i];
		DC[i] = D[i] - C[i];
	    }
	    ofmo_twoint_core_os_dddd( &La, &Lb, &Lc, &Ld,
		    &nijps, &psp_zeta_frg[ijps0], &psp_dkps_frg[ijps0],
		    &psp_xiza_frg[ijps0], BA,
		    &nklps, &psp_zeta_mon[klps0], &psp_dkps_mon[klps0],
		    &psp_xiza_mon[klps0], DC,   AC,      DDDD );
	    for ( i=0, iao=iao0, ix=0; i<6; i++, iao++ ) {
		I2 = (iao*iao+iao)>>1;
		for ( j=0, jao=jao0; j<6; j++, jao++ ) {
		    if ( jao>iao ) { ix+=6*6; continue; }
		    IJ = I2 + jao;
		    for ( k=0, kao=kao0; k<6; k++, kao++ ) {
			K2 = (kao*kao+kao)>>1;
			for ( l=0, lao=lao0; l<6; l++, lao++, ix++ ) {
			    if ( lao>kao ) continue;
			    coe = (kao==lao? ONE : TWO );
			    KL = K2 + lao;
			    if ( fabs(DDDD[ix]) > eps_eri ) {
				V_frg[IJ] += coe * D_mon[KL] * DDDD[ix];
			    }
			}
		    }
		}
	    }
	}	// klcs
    }		// ijcs
    return 0;
}	// end of ofmo_ifc4c_dddd_

#endif /* if 0 */

#undef NREGS_0
#undef NREGS_1
#undef NREGS_2
#undef NREGS_3
#undef NREGS_4

#undef CNV_CSP
