#ifndef MILUPHPC_KERNEL_CUH
#define MILUPHPC_KERNEL_CUH

#include "../particles.cuh"
#include "../parameter.h"
#include "../helper.cuh"

//#include <boost/mpi.hpp>
#include <assert.h>
#include "../parameter.h"
#include "../cuda_utils/linalg.cuh"

namespace SPH {


    typedef void (*SPH_kernel)(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real h);

    namespace SmoothingKernel {

        __device__ void spiky(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);

        __device__ void cubicSpline(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);

        __device__ void wendlandc2(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);

        __device__ void wendlandc4(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);

        __device__ void wendlandc6(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);
    }


    //TODO: implement:
    CUDA_CALLABLE_MEMBER real fixTensileInstability(SPH_kernel kernel, Particles *particles, int p1, int p2);

#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)
    __global__ void CalcDivvandCurlv(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles);
#endif

#if ZERO_CONSISTENCY //SHEPARD_CORRECTION
    // this adds zeroth order consistency but needs one more loop over all neighbours
    __global__ void shepardCorrection(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles)
#endif

#if LINEAR_CONSISTENCY //TENSORIAL_CORRECTION
// this adds first order consistency but needs one more loop over all neighbours
__global__ void tensorialCorrection(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles)
#endif

}


#endif //MILUPHPC_KERNEL_CUH
