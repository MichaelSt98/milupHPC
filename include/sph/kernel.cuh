#ifndef MILUPHPC_KERNEL_CUH
#define MILUPHPC_KERNEL_CUH

#include "../particles.cuh"
#include "../parameter.h"
#include "../helper.cuh"

//#include <boost/mpi.hpp>
#include <assert.h>

class Kernel {

public:
    virtual CUDA_CALLABLE_MEMBER void kernel(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml) = 0;

    //TODO: implement:
    CUDA_CALLABLE_MEMBER real fixTensileInstability(Particles *particles, int p1, int p2);
    //__global__ void tensorialCorrection(int *interactions);
    //__global__ void shepardCorrection(int *interactions);
    //__global__ void CalcDivvandCurlv(int *interactions);

};

class Spiky : public Kernel {

public:
    CUDA_CALLABLE_MEMBER void kernel(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);

};

class CubicSpline : public Kernel {

public:
    CUDA_CALLABLE_MEMBER void kernel(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);
};


#endif //MILUPHPC_KERNEL_CUH
