#include "cuda-twoint-core-rys.h"
#include "cuda-root.h"

__device__ void gpu_hrr_clear_rys_ppps( double *eh ) {
    int i;
    // (PS|PS)
#pragma unroll
    for ( i=0; i<(0+9); i++ ) eh[i] = 0.e0;
    // (DS|PS)
#pragma unroll
    for ( i=9; i<(9+18); i++ ) eh[i] = 0.e0;
}

__device__ void gpu_hrr_coef_rys_ppps(
        double *eh, double *DINT ) {
    int i, j, k, l, iao, jao, kao, lao, ix;
    double coef_a, coef_ab, coef_abc;
    double *th;
    th = &eh[27];
    ix = 0;
#pragma unroll
    for ( i=0, iao=1; i<3; i++, iao++ ) {
        coef_a = DFACT[iao];
#pragma unroll
        for ( j=0, jao=1; j<3; j++, jao++ ) {
            coef_ab = coef_a * DFACT[jao];
#pragma unroll
            for ( k=0, kao=1; k<3; k++, kao++ ) {
                coef_abc = coef_ab * DFACT[kao];
#pragma unroll
                for ( l=0, lao=0; l<1; l++, lao++ ) {
                    DINT[ix] = coef_abc * DFACT[lao] * th[ix];
                    ix++;
                }
            }
        }
    }
}

__device__ void gpu_hrr_calc_rys_ppps( double *eh,
        const double BA[3], const double DC[3] ) {
    // HRR for (XX|XS)-type integral (center AB)
    // (PP,PS)
    eh[  27] = eh[   9] - BA[0]*eh[   0];
    eh[  28] = eh[  10] - BA[0]*eh[   1];
    eh[  29] = eh[  11] - BA[0]*eh[   2];
    eh[  30] = eh[  18] - BA[1]*eh[   0];
    eh[  31] = eh[  19] - BA[1]*eh[   1];
    eh[  32] = eh[  20] - BA[1]*eh[   2];
    eh[  33] = eh[  21] - BA[2]*eh[   0];
    eh[  34] = eh[  22] - BA[2]*eh[   1];
    eh[  35] = eh[  23] - BA[2]*eh[   2];
    eh[  36] = eh[  18] - BA[0]*eh[   3];
    eh[  37] = eh[  19] - BA[0]*eh[   4];
    eh[  38] = eh[  20] - BA[0]*eh[   5];
    eh[  39] = eh[  12] - BA[1]*eh[   3];
    eh[  40] = eh[  13] - BA[1]*eh[   4];
    eh[  41] = eh[  14] - BA[1]*eh[   5];
    eh[  42] = eh[  24] - BA[2]*eh[   3];
    eh[  43] = eh[  25] - BA[2]*eh[   4];
    eh[  44] = eh[  26] - BA[2]*eh[   5];
    eh[  45] = eh[  21] - BA[0]*eh[   6];
    eh[  46] = eh[  22] - BA[0]*eh[   7];
    eh[  47] = eh[  23] - BA[0]*eh[   8];
    eh[  48] = eh[  24] - BA[1]*eh[   6];
    eh[  49] = eh[  25] - BA[1]*eh[   7];
    eh[  50] = eh[  26] - BA[1]*eh[   8];
    eh[  51] = eh[  15] - BA[2]*eh[   6];
    eh[  52] = eh[  16] - BA[2]*eh[   7];
    eh[  53] = eh[  17] - BA[2]*eh[   8];
}

__device__ void gpu_xyzint_rys_ppps(
        const double *F00, const double *B00, const double *B10,
        const double *B01, const double *C00, const double *CP00,
        double *xint, double *yint, double *zint ) {
    int Lab, Lcd;
    int m, m3, N, M, ix3, ix2, ix1, ix0, nroot;
    double C10[2], CP10[2], CP01[2], C01[2];
    // (0,0)
    xint[  0]=1.e0;
    yint[  0]=1.e0;
    zint[  0]=F00[0];
    xint[  1]=1.e0;
    yint[  1]=1.e0;
    zint[  1]=F00[1];
    // (1,0)
    xint[  4]=C00[ 0];
    yint[  4]=C00[ 1];
    zint[  4]=C00[ 2]*F00[0];
    xint[  5]=C00[ 3];
    yint[  5]=C00[ 4];
    zint[  5]=C00[ 5]*F00[1];
    // (0,1)
    xint[  2]=CP00[ 0];
    yint[  2]=CP00[ 1];
    zint[  2]=CP00[ 2]*F00[0];
    xint[  3]=CP00[ 3];
    yint[  3]=CP00[ 4];
    zint[  3]=CP00[ 5]*F00[1];
    // (1,1)
    xint[  6]=CP00[ 0]*xint[  4]+B00[0];
    yint[  6]=CP00[ 1]*yint[  4]+B00[0];
    zint[  6]=CP00[ 2]*zint[  4]+B00[0]*F00[0];
    xint[  7]=CP00[ 3]*xint[  5]+B00[1];
    yint[  7]=CP00[ 4]*yint[  5]+B00[1];
    zint[  7]=CP00[ 5]*zint[  5]+B00[1]*F00[1];
    // (N,0) and (N,1)
#pragma unroll
    for ( m=0; m<2; m++ ) {
        C10[m]  = 0.e0;
        CP10[m] = B00[m];
    }
    // (2,0)
    C10[0] += B10[0];
    xint[  8]=C00[ 0]*xint[  4]+C10[0]*xint[  0];
    yint[  8]=C00[ 1]*yint[  4]+C10[0]*yint[  0];
    zint[  8]=C00[ 2]*zint[  4]+C10[0]*zint[  0];
    C10[1] += B10[1];
    xint[  9]=C00[ 3]*xint[  5]+C10[1]*xint[  1];
    yint[  9]=C00[ 4]*yint[  5]+C10[1]*yint[  1];
    zint[  9]=C00[ 5]*zint[  5]+C10[1]*zint[  1];
    // (2,1)
    CP10[0] += B00[0];
    xint[ 10]=CP00[ 0]*xint[  8]+CP10[0]*xint[  4];
    yint[ 10]=CP00[ 1]*yint[  8]+CP10[0]*yint[  4];
    zint[ 10]=CP00[ 2]*zint[  8]+CP10[0]*zint[  4];
    CP10[1] += B00[1];
    xint[ 11]=CP00[ 3]*xint[  9]+CP10[1]*xint[  5];
    yint[ 11]=CP00[ 4]*yint[  9]+CP10[1]*yint[  5];
    zint[ 11]=CP00[ 5]*zint[  9]+CP10[1]*zint[  5];
}

__device__ void gpu_form_rys_ppps(
        const double *xint, const double *yint, const double *zint,
        double *eh ) {
    // (PS|PS)
    eh[   0] += xint[  6]*yint[  0]*zint[  0];
    eh[   0] += xint[  7]*yint[  1]*zint[  1];
    eh[   1] += xint[  4]*yint[  2]*zint[  0];
    eh[   1] += xint[  5]*yint[  3]*zint[  1];
    eh[   2] += xint[  4]*yint[  0]*zint[  2];
    eh[   2] += xint[  5]*yint[  1]*zint[  3];
    eh[   3] += xint[  2]*yint[  4]*zint[  0];
    eh[   3] += xint[  3]*yint[  5]*zint[  1];
    eh[   4] += xint[  0]*yint[  6]*zint[  0];
    eh[   4] += xint[  1]*yint[  7]*zint[  1];
    eh[   5] += xint[  0]*yint[  4]*zint[  2];
    eh[   5] += xint[  1]*yint[  5]*zint[  3];
    eh[   6] += xint[  2]*yint[  0]*zint[  4];
    eh[   6] += xint[  3]*yint[  1]*zint[  5];
    eh[   7] += xint[  0]*yint[  2]*zint[  4];
    eh[   7] += xint[  1]*yint[  3]*zint[  5];
    eh[   8] += xint[  0]*yint[  0]*zint[  6];
    eh[   8] += xint[  1]*yint[  1]*zint[  7];
    // (DS|PS)
    eh[   9] += xint[ 10]*yint[  0]*zint[  0];
    eh[   9] += xint[ 11]*yint[  1]*zint[  1];
    eh[  10] += xint[  8]*yint[  2]*zint[  0];
    eh[  10] += xint[  9]*yint[  3]*zint[  1];
    eh[  11] += xint[  8]*yint[  0]*zint[  2];
    eh[  11] += xint[  9]*yint[  1]*zint[  3];
    eh[  12] += xint[  2]*yint[  8]*zint[  0];
    eh[  12] += xint[  3]*yint[  9]*zint[  1];
    eh[  13] += xint[  0]*yint[ 10]*zint[  0];
    eh[  13] += xint[  1]*yint[ 11]*zint[  1];
    eh[  14] += xint[  0]*yint[  8]*zint[  2];
    eh[  14] += xint[  1]*yint[  9]*zint[  3];
    eh[  15] += xint[  2]*yint[  0]*zint[  8];
    eh[  15] += xint[  3]*yint[  1]*zint[  9];
    eh[  16] += xint[  0]*yint[  2]*zint[  8];
    eh[  16] += xint[  1]*yint[  3]*zint[  9];
    eh[  17] += xint[  0]*yint[  0]*zint[ 10];
    eh[  17] += xint[  1]*yint[  1]*zint[ 11];
    eh[  18] += xint[  6]*yint[  4]*zint[  0];
    eh[  18] += xint[  7]*yint[  5]*zint[  1];
    eh[  19] += xint[  4]*yint[  6]*zint[  0];
    eh[  19] += xint[  5]*yint[  7]*zint[  1];
    eh[  20] += xint[  4]*yint[  4]*zint[  2];
    eh[  20] += xint[  5]*yint[  5]*zint[  3];
    eh[  21] += xint[  6]*yint[  0]*zint[  4];
    eh[  21] += xint[  7]*yint[  1]*zint[  5];
    eh[  22] += xint[  4]*yint[  2]*zint[  4];
    eh[  22] += xint[  5]*yint[  3]*zint[  5];
    eh[  23] += xint[  4]*yint[  0]*zint[  6];
    eh[  23] += xint[  5]*yint[  1]*zint[  7];
    eh[  24] += xint[  2]*yint[  4]*zint[  4];
    eh[  24] += xint[  3]*yint[  5]*zint[  5];
    eh[  25] += xint[  0]*yint[  6]*zint[  4];
    eh[  25] += xint[  1]*yint[  7]*zint[  5];
    eh[  26] += xint[  0]*yint[  4]*zint[  6];
    eh[  26] += xint[  1]*yint[  5]*zint[  7];
}

__device__ void gpu_twoint_core_rys_ppps(
        const int *nijps, const double *vzeta, const double *vdkab,
        const double vxiza[], const double BA[3],
        const int *nklps, const double *veta, const double *vdkcd,
        const double *vxizc, const double DC[3], const double AC[3],
        double *DINT ) {
    int ijps, klps, i;
    double cssss, zeta, dkab, xiza, eta, xizc, dk, T;
    double zeta2, eta2, rz, PA[3], QC[3];
    double PQ2, sqrho, rho, PC[3], QP[3];
    double C00[6], CP00[6], B00[2], B10[2], B01[2], F00[2];
    //double rrho, rze, W[13], U[13];
    double rrho, rze, W[2], U[2];
    double u2, duminv, dm2inv, dum;
    int m, m3;
    /*
    double *xint, *yint, *zint, *eh;
    xint = gpu_integ_getadd_xint( mythread );
    yint = gpu_integ_getadd_yint( mythread );
    zint = gpu_integ_getadd_zint( mythread );
    eh   = gpu_integ_getadd_eh( mythread );
    */
    double xint[12], yint[12], zint[12];
    double eh[54];

    gpu_hrr_clear_rys_ppps( eh );
    for ( ijps=0; ijps<(*nijps); ijps++ ) {
        zeta  = vzeta[ijps];
        dkab  = vdkab[ijps];
        xiza  = vxiza[ijps];
        zeta2 = HALF * zeta;
#pragma unroll
        for ( i=0; i<3; i++ ) {
            PC[i] = AC[i] + xiza*BA[i];
            PA[i] = xiza * BA[i];
        }
        for ( klps=0; klps<(*nklps); klps++ ) {
            eta  = veta[klps];
            dk   = dkab * vdkcd[klps];
            xizc = vxizc[klps];
            eta2 = HALF * eta;
            PQ2  = ZERO;
#pragma unroll
            for ( i=0; i<3; i++ ) {
                QC[i] = xizc*DC[i];
                QP[i] = xizc*DC[i] - PC[i];
                PQ2  += QP[i]*QP[i];
            }
            rrho  = zeta + eta;
            rze   = zeta * eta;
            sqrho = sqrt(1.e0/rrho);
            rho   = sqrho * sqrho;
            rz    = rho * zeta;
            T     = rho * PQ2;
            cssss = sqrho * dk;
            gpu_calc_root( 2, T, U, W );
#pragma unroll
            for ( m=m3=0; m<2; m++, m3+=3 ) {
                u2     = rho * U[m];
                F00[m] = cssss * W[m];
                duminv = 1.e0 / ( 1.e0 + rrho * u2 );
                dm2inv = 0.5e0 * duminv;
                B00[m] = dm2inv * rze * u2;
                B10[m] = dm2inv * ( zeta + rze*u2 );
                B01[m] = dm2inv * ( eta  + rze*u2 );
                dum    = zeta * u2 * duminv;
#pragma unroll
                for ( i=0; i<3; i++ ) C00[m3+i]  = PA[i] + dum * QP[i];
                dum    = eta * u2 * duminv;
#pragma unroll
                for ( i=0; i<3; i++ ) CP00[m3+i] = QC[i] - dum * QP[i];
            }
            gpu_xyzint_rys_ppps( F00, B00, B10, B01, C00, CP00,
                     xint, yint, zint );
            gpu_form_rys_ppps( xint, yint, zint, eh );
        }
    }
    gpu_hrr_calc_rys_ppps( eh, BA, DC );
    gpu_hrr_coef_rys_ppps( eh, DINT );
}

#if 0
int gpu_twoint_rys_ppps(
        const int *pnworkers, const int *pworkerid,
        const int *pLa, const int *pLb, const int *pLc, const int *pLd,
        const int *shel_atm, const int *shel_ini,
        const double atom_x[], const double atom_y[],
        const double atom_z[], const int leading_cs_pair[],
        const double csp_schwarz[],
        const int csp_ics[], const int csp_jcs[],
        const int csp_leading_ps_pair[],
        const double psp_zeta[], const double psp_dkps[],
        const double psp_xiza[],
        // for partially direct SCF
        const long *pebuf_max_nzeri, long *ebuf_non_zero_eri,
        double ebuf_val[], short int ebuf_ind4[],
        int *last_ijcs, int *last_klcs ) {
    int Lab, Lcd, i, j, k, l, ipat, ix;
    int I2, IJ, K2, KL;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, max_klcs;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe, coe0;
    double *DINTEG;
    long nzeri, max_nzeri, nzeri4;
    int nworkers=*pnworkers, workerid=*pworkerid;
    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    long ebuf_max_nzeri = *pebuf_max_nzeri;
    int mythread;
    DFACT = gpu_getadd_dfact();
    mythread = omp_get_thread_num();
    DINTEG = gpu_integ_getadd_eri( mythread );
    Lab = La*(La+1)/2+Lb;
    Lcd = Lc*(Lc+1)/2+Ld;
    ijcs0 = leading_cs_pair[Lab];
    ijcs1 = leading_cs_pair[Lab+1];
    klcs0 = leading_cs_pair[Lcd];
    klcs1 = leading_cs_pair[Lcd+1];
    nzeri     = *ebuf_non_zero_eri;
    max_nzeri = ebuf_max_nzeri - 3*3*3*1;
    nzeri4    = nzeri*4;
    if ( nzeri >= max_nzeri ) {
        *last_ijcs = ijcs0+workerid;
        *last_klcs = klcs0 - 1;
        *ebuf_non_zero_eri = nzeri;
        return OFMO_EBUF_FULL;
    }
    for ( ijcs=ijcs0+workerid; ijcs<ijcs1; ijcs+=nworkers ) {
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
        max_klcs = ( Lab == Lcd ? ijcs+1 : klcs1 );
        for ( klcs=klcs0; klcs<max_klcs; klcs++ ) {
            val_cd = csp_schwarz[klcs];
            if ( val_ab*val_cd < EPS_PS4 ) continue;
            kcs    = csp_ics[klcs];
            lcs    = csp_jcs[klcs];
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
            gpu_twoint_core_rys_ppps( mythread,
                    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
                    &psp_xiza[ijps0], BA,
                    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
                    &psp_xiza[klps0], DC,   AC,      DINTEG );
            ipat=((Lab != Lcd) || (ics==kcs && jcs>lcs) ? true : false);
            for ( i=0, iao=iao0, ix=0; i<3; i++, iao++ ) {
                I2 = (iao*iao+iao)>>1;
                for ( j=0, jao=jao0; j<3; j++, jao++ ) {
                    if ( jao>iao ) { ix+=3*1; continue; }
                    IJ = I2 + jao;
                    coe0 = ( iao==jao ? HALF : ONE );
                    for ( k=0, kao=kao0; k<3; k++, kao++ ) {
                        K2 = (kao*kao+kao)>>1;
                        for ( l=0, lao=lao0; l<1; l++, lao++, ix++ ) {
                            if ( lao>kao ) continue;
                            if ( fabs(DINTEG[ix]) > EPS_ERI ) {
                                KL = K2 + lao;
                                if ( IJ >= KL ) {
                                    coe = coe0;
                                    if ( kao==lao ) coe *= HALF;
                                    if ( KL == IJ ) coe *= HALF;
                                    ebuf_val[nzeri]     = coe*DINTEG[ix];
                                    ebuf_ind4[nzeri4+0] = (short int)iao;
                                    ebuf_ind4[nzeri4+1] = (short int)jao;
                                    ebuf_ind4[nzeri4+2] = (short int)kao;
                                    ebuf_ind4[nzeri4+3] = (short int)lao;
                                    nzeri++;
                                    nzeri4+=4;
                                } else if ( ipat ) {
                                    coe = coe0;
                                    if ( kao==lao ) coe*=HALF;
                                    ebuf_val[nzeri]     = coe*DINTEG[ix];
                                    ebuf_ind4[nzeri4+0] = (short int)kao;
                                    ebuf_ind4[nzeri4+1] = (short int)lao;
                                    ebuf_ind4[nzeri4+2] = (short int)iao;
                                    ebuf_ind4[nzeri4+3] = (short int)jao;
                                    nzeri++;
                                    nzeri4+=4;
                                }
                            }
                        }
                    }
                }
            }
            if ( nzeri >= max_nzeri ) {
                *last_ijcs = ijcs;
                *last_klcs = klcs;
                *ebuf_non_zero_eri = nzeri;
                return OFMO_EBUF_FULL;
            }
        }
    }
    *ebuf_non_zero_eri = nzeri;
    return OFMO_EBUF_NOFULL;
}

int gpu_twoint_direct_rys_ppps(
        const int *pnworkers, const int *pworkerid,
        const int *pLa, const int *pLb, const int *pLc, const int *pLd,
        const int *shel_atm, const int *shel_ini,
        const double atom_x[], const double atom_y[],
        const double atom_z[], const int leading_cs_pair[],
        const double csp_schwarz[],
        const int csp_ics[], const int csp_jcs[],
        const int csp_leading_ps_pair[],
        const double psp_zeta[], const double psp_dkps[],
        const double psp_xiza[],
        // for direct SCF
        const long *petmp_max_nzeri, long *petmp_non_zero_eri,
        double etmp_val[], short int etmp_ind4[],
        const int *plast_ijcs, const int *plast_klcs,
        // density matrix & G-matrix data
        const int *pnao, const double Ds[], double G[] ) {
    int nworkers=*pnworkers, workerid=*pworkerid;
    int La=*pLa, Lb=*pLb, Lc=*pLc, Ld=*pLd;
    int last_ijcs=*plast_ijcs, last_klcs=*plast_klcs, nao=*pnao;
    long max_nzeri=*petmp_max_nzeri;
    long nzeri4, nzeri=*petmp_non_zero_eri;
    //
    int Lab, Lcd, i, j, k, l, ipat, ix;
    int I2, IJ, K2, KL;
    int ijcs, ijcs0, ijcs1;
    int klcs, klcs0, klcs1, max_klcs;
    int ijps0, nijps, klps0, nklps;
    int ics, iat, iao, iao0, jcs, jat, jao, jao0;
    int kcs, kat, kao, kao0, lcs, lat, lao, lao0;
    double A[3], B[3], C[3], D[3], BA[3], DC[3], AC[3];
    double val_ab, val_cd, coe, coe0;
    double *DINTEG;
    int mythread;
    DFACT = gpu_getadd_dfact();
    mythread = omp_get_thread_num();
    DINTEG = gpu_integ_getadd_eri( mythread );
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
    max_nzeri -= 3*3*3*1;
    nzeri4    = nzeri*4;
    if ( nzeri >= max_nzeri ) {
        gpu_integ_add_fock( nao, nzeri, etmp_val, etmp_ind4, Ds, G );
        nzeri = nzeri4 = 0;
    }
    for ( ; ijcs<ijcs1; ijcs+=nworkers ) {
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
        max_klcs = ( Lab == Lcd ? ijcs+1 : klcs1 );
        for ( ; klcs<max_klcs; klcs++ ) {
            val_cd = csp_schwarz[klcs];
            if ( val_ab*val_cd < EPS_PS4 ) continue;
            kcs    = csp_ics[klcs];
            lcs    = csp_jcs[klcs];
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
            gpu_twoint_core_rys_ppps( mythread,
                    &nijps, &psp_zeta[ijps0], &psp_dkps[ijps0],
                    &psp_xiza[ijps0], BA,
                    &nklps, &psp_zeta[klps0], &psp_dkps[klps0],
                    &psp_xiza[klps0], DC,   AC,      DINTEG );
            ipat=((Lab != Lcd) || (ics==kcs && jcs>lcs) ? true : false);
            for ( i=0, iao=iao0, ix=0; i<3; i++, iao++ ) {
                I2 = (iao*iao+iao)>>1;
                for ( j=0, jao=jao0; j<3; j++, jao++ ) {
                    if ( jao>iao ) { ix+=3*1; continue; }
                    IJ = I2 + jao;
                    coe0 = ( iao==jao ? HALF : ONE );
                    for ( k=0, kao=kao0; k<3; k++, kao++ ) {
                        K2 = (kao*kao+kao)>>1;
                        for ( l=0, lao=lao0; l<1; l++, lao++, ix++ ) {
                            if ( lao>kao ) continue;
                            if ( fabs(DINTEG[ix]) > EPS_ERI ) {
                                KL = K2 + lao;
                                if ( IJ >= KL ) {
                                    coe = coe0;
                                    if ( kao==lao ) coe *= HALF;
                                    if ( KL == IJ ) coe *= HALF;
                                    etmp_val[nzeri]     = coe*DINTEG[ix];
                                    etmp_ind4[nzeri4+0] = (short int)iao;
                                    etmp_ind4[nzeri4+1] = (short int)jao;
                                    etmp_ind4[nzeri4+2] = (short int)kao;
                                    etmp_ind4[nzeri4+3] = (short int)lao;
                                    nzeri++;
                                    nzeri4+=4;
                                } else if ( ipat ) {
                                    coe = coe0;
                                    if ( kao==lao ) coe*=HALF;
                                    etmp_val[nzeri]     = coe*DINTEG[ix];
                                    etmp_ind4[nzeri4+0] = (short int)kao;
                                    etmp_ind4[nzeri4+1] = (short int)lao;
                                    etmp_ind4[nzeri4+2] = (short int)iao;
                                    etmp_ind4[nzeri4+3] = (short int)jao;
                                    nzeri++;
                                    nzeri4+=4;
                                }
                            }
                        }
                    }
                }
            }
            if ( nzeri >= max_nzeri ) {
                gpu_integ_add_fock( nao, nzeri, etmp_val, etmp_ind4,
                        Ds, G );
                nzeri = nzeri4= 0;
            }
        }
        klcs = klcs0;
    }
    *petmp_non_zero_eri = nzeri;
    return 0;
}
#endif
