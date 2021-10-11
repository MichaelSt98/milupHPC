#ifndef MILUPHPC_LINALG_CUH
#define MILUPHPC_LINALG_CUH

#include "../parameter.h"

class linalg {

};

namespace CudaUtils {

    __device__ void copyMatrix(double src[DIM][DIM], double dst[DIM][DIM]);

    __device__ void transposeMatrix(double m[DIM][DIM]);

    __device__  void multiplyMatrix(double A[DIM][DIM], double B[DIM][DIM], double C[DIM][DIM]);

    __device__ void identityMatrix(double A[DIM][DIM]);

    __device__ int maxMatrix(double M[DIM][DIM], int *e, int *f, double *elmax);

    __device__ void rotateMatrix(volatile double m[DIM][DIM], volatile double c, volatile double s, volatile int e,
                                  volatile int f);

    __device__ int calculateAllEigenvalues(double M[DIM][DIM], double eigenvalues[DIM], double v[DIM][DIM]);

    __device__ double calculateMaxEigenvalue(double M[DIM][DIM]);

    __device__ double det2x2(double a, double b, double c, double d);

    __device__ int invertMatrix(double *m, double *inverted);
}


#endif //MILUPHPC_LINALG_CUH
