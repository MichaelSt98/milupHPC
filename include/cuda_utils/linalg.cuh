/**
 * @file linalg.cuh
 * @brief Linear algebra CUDA kernels and device functions.
 *
 * More detailed information.
 *
 * @author Michael Staneker
 * @bug no
*/
#ifndef MILUPHPC_LINALG_CUH
#define MILUPHPC_LINALG_CUH

#include "../parameter.h"
#include "../../include/cuda_utils/cuda_utilities.cuh"

#define FLOAT_ZERO_TOLERANCE 1e-12

class linalg {

};

namespace CudaUtils {

    /**
     * @brief Get sign of floating point variable.
     *
     * @param x Floating point variable
     * @return Sign of floating point variable (-1, 0, 1)
     */
    __device__ int sign(real x);

    /**
     * @brief map `[i][j]` to `[i*DIM*DIM+j]` for the tensors
     */
    __device__ int stressIndex(int particleIndex, int row, int col);

    /**
     * @brief Deep copy of matrix.
     *
     * @param[in] src Source matrix
     * @param[out] dst Destination matrix
     */
    __device__ void copyMatrix(real src[DIM][DIM], real dst[DIM][DIM]);

    /**
     * @brief Transpose matrix.
     *
     * @param[in, out] m Matrix to be transposed
     */
    __device__ void transposeMatrix(real m[DIM][DIM]);

    /**
     * Multiply two matrices.
     *
     * @param[in] A Matrix `A`
     * @param[in] B Matrix `B`
     * @param[out] C Result matrix
     */
    __device__  void multiplyMatrix(real A[DIM][DIM], real B[DIM][DIM], real C[DIM][DIM]);

    /**
     * Multiply.
     *
     * @param[in] A
     * @param[in] B
     * @param[out] C
     */
    __device__ void multiply(real A[][DIM], real B[][DIM], real C[][DIM]);

    /**
     * @brief Return identity matrix.
     *
     * @param[in, out] A Identity matrix.
     */
    __device__ void identityMatrix(real A[DIM][DIM]);

    /**
     * @brief Returns the indices of the greatest non-diagonal element of the matrix `M`.
     *
     * @param M Relevant matrix
     * @param e
     * @param f
     * @param elmax
     * @return error code
     */
    __device__ int maxMatrix(real M[DIM][DIM], int *e, int *f, real *elmax);

    /**
     * @brief Rotate matrix.
     *
     * Returns: M' = A^T M A, and A_ef = s = -A_ef, A_ee = A_ff = c
     *
     * @param m
     * @param c
     * @param s
     * @param e
     * @param f
     */
    __device__ void rotateMatrix(volatile real m[DIM][DIM], volatile real c, volatile real s, volatile int e,
                                  volatile int f);

    /**
     * @brief Computes all eigenvalues and eigenvectors of the symmetric matrix `M`.
     *
     * using the jacobi method and stores them in `eigenvals` and the `eigenvecs` as columns
     * in the transformation matrix v
     *
     * @param M
     * @param eigenvalues
     * @param v
     * @return number of iterations
     */
    __device__ int calculateAllEigenvalues(real M[DIM][DIM], real eigenvalues[DIM], real v[DIM][DIM]);

    /**
     * @brief Computes the eigenvalues of the symmetric matrix `M`.
     *
     * Using the jacobi method.
     *
     * @param M
     * @return Greatest eigenvalue
     */
    __device__ real calculateMaxEigenvalue(real M[DIM][DIM]);

    /**
     * @brief Determinant of a 2x2 matrix.
     *
     * @param a
     * @param b
     * @param c
     * @param d
     * @return
     */
    __device__ real det2x2(real a, real b, real c, real d);

    /**
     * @brief Invert matrix.
     *
     * @param[in] m
     * @param[out] inverted
     * @return
     */
    __device__ int invertMatrix(real *m, real *inverted);
}


#endif //MILUPHPC_LINALG_CUH
