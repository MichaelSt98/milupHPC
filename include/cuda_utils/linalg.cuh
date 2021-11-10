#ifndef MILUPHPC_LINALG_CUH
#define MILUPHPC_LINALG_CUH

#include "../parameter.h"
#include "../../include/cuda_utils/cuda_utilities.cuh"

class linalg {

};

namespace CudaUtils {

    __device__ int sign(real x);

    // map [i][j] to [i*DIM*DIM+j] for the tensors
    __device__ int stressIndex(int particleIndex, int row, int col);

    __device__ void copyMatrix(real src[DIM][DIM], real dst[DIM][DIM]);

    __device__ void transposeMatrix(real m[DIM][DIM]);

    __device__  void multiplyMatrix(real A[DIM][DIM], real B[DIM][DIM], real C[DIM][DIM]);
    __device__ void multiply(real A[][DIM], real B[][DIM], real C[][DIM]);

    __device__ void identityMatrix(real A[DIM][DIM]);

    __device__ int maxMatrix(real M[DIM][DIM], int *e, int *f, real *elmax);

    __device__ void rotateMatrix(volatile real m[DIM][DIM], volatile real c, volatile real s, volatile int e,
                                  volatile int f);

    __device__ int calculateAllEigenvalues(real M[DIM][DIM], real eigenvalues[DIM], real v[DIM][DIM]);

    __device__ real calculateMaxEigenvalue(real M[DIM][DIM]);

    __device__ real det2x2(real a, real b, real c, real d);

    __device__ int invertMatrix(real *m, real *inverted);
}


#endif //MILUPHPC_LINALG_CUH
