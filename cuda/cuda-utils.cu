#include "cuda-utils.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
// From cuda toolkit documentation 
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


/* --------------------------------------------------- */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
//#if CUDA_ARCH >= 350

__device__ double warpReduceG(double v, const int tid)
{
  v += __shfl_down_sync(0xffffffff, v, 16, 32);
  v += __shfl_down_sync(0xffffffff, v,  8, 32);
  v += __shfl_down_sync(0xffffffff, v,  4, 32);
  v += __shfl_down_sync(0xffffffff, v,  2, 32);
  v += __shfl_down_sync(0xffffffff, v,  1, 32);
  return v;
}

#endif

__device__ void warpReduce(volatile double *sdata, const int tid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
//#if CUDA_ARCH >= 350
  double v = sdata[tid];
  sdata[tid] = warpReduceG(v, tid);
#else
  if (tid<WARP_SIZE/2) {
//  sdata[tid] += sdata[tid+32];
  sdata[tid] += sdata[tid+16];
  __syncwarp();
  sdata[tid] += sdata[tid+8];
  __syncwarp();
  sdata[tid] += sdata[tid+4];
  __syncwarp();
  sdata[tid] += sdata[tid+2];
  __syncwarp();
  sdata[tid] += sdata[tid+1];
  __syncwarp();
  }
#endif
}

