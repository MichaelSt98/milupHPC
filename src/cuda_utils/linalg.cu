#include "../../include/cuda_utils/linalg.cuh"

namespace CudaUtils {

    __device__ int sign(real x) {
        if (x < 0) { return -1; }
        else if (x > 0) { return  1; }
        else { return 0; }
    }

    __device__ int stressIndex(int particleIndex, int row, int col)
    {
        return particleIndex*DIM*DIM+row*DIM+col;
    }

    __device__ void copyMatrix(real src[DIM][DIM], real dst[DIM][DIM]) {
        int i, j;

        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                dst[i][j] = src[i][j];
            }
        }

    }

    __device__ void transposeMatrix(real m[DIM][DIM]) {
        int i, j;
        real mt[DIM][DIM];
        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                mt[j][i] = m[i][j];
            }
        }
        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                m[i][j] = mt[i][j];
            }
        }
    }

    // calculates C = A B and stores in C
    __device__  void multiplyMatrix(real A[DIM][DIM], real B[DIM][DIM], real C[DIM][DIM]) {
        int i, j, k;

        real vprime[DIM][DIM];

        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                vprime[i][j] = 0.0;
            }
        }

        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                for (k = 0; k < DIM; k++) {
                    vprime[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                C[i][j] = vprime[i][j];
            }
        }

    }

    __device__ void multiply(real A[][DIM], real B[][DIM], real C[][DIM]) {
        int i, j, k;
        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                for (k = 0; k < DIM; k++) {
                    C[i][j] += A[i][k]*B[k][j];
                }
            }
        }
    }

    __device__ void identityMatrix(real A[DIM][DIM]) {
        int i, j;
        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                A[i][j] = 0.0;
            }
            A[i][i] = 1.0;
        }
    }


    // returns the indices of the greatest non-diagonal element of M
    __device__ int maxMatrix(real M[DIM][DIM], int *e, int *f, real *elmax) {
        int i, j;
        real max = 0.0;
        int ierror = 1;

        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                if (i == j)
                    continue;
                if (cuda::math::abs(M[i][j]) >= max) {
                    max = cuda::math::abs(M[i][j]);
                    *e = i;
                    *f = j;
                    ierror = 0;
                }
            }
        }
        *elmax = max;
        return ierror;
    }


    //
    // help function for the jacobi method
    // returns: M' = A^T M A, and A_ef = s = -A_ef, A_ee = A_ff = c
    //
    __device__ void rotateMatrix(volatile real m[DIM][DIM], volatile real c, volatile real s, volatile int e,
                                  volatile int f) {
        int i, j;
        volatile real mprime[DIM][DIM];

        // first copy the matrix
        for (i = 0; i < DIM; i++)
            for (j = 0; j < DIM; j++)
                mprime[i][j] = m[i][j];

        // now the elements that change
        mprime[e][e] = c * c * m[e][e] + s * s * m[f][f] - 2 * s * c * m[e][f];
        mprime[f][f] = c * c * m[f][f] + s * s * m[e][e] + 2 * s * c * m[e][f];
        mprime[e][f] = (c * c - s * s) * m[e][f] + s * c * (m[e][e] - m[f][f]);
        mprime[f][e] = mprime[e][f];

        // the other elements in columns and rows e, f
        // actually, this is only one in 3D and 0 in 2D
        for (i = 0; i < DIM; i++) {
            if (i == f || i == e)
                continue;
            mprime[e][i] = c * m[i][e] - s * m[i][f];
            mprime[i][e] = mprime[e][i];
            mprime[f][i] = c * m[i][f] + s * m[i][e];
            mprime[i][f] = mprime[f][i];
        }

        // set the matrix to the rotated one
        for (i = 0; i < DIM; i++)
            for (j = 0; j < DIM; j++)
                m[i][j] = mprime[i][j];
    }


    //
    // computes all eigenvalues and eigenvectors of the _symmetric_ matrix M
    // using the jacobi method and stores them in eigenvals and the eigenvecs as columns
    // in the transformation matrix v
    //
    // returns the number of iterations
    //
    __device__ int calculateAllEigenvalues(real M[DIM][DIM], real eigenvalues[DIM], real v[DIM][DIM]) {
        int i, j;
        real diagM[DIM][DIM] = {0.0,};
        real c, s, t, thta;
        real A[DIM][DIM];
        real vtmp[DIM][DIM];
        int e, f;
        int error;
        real max = -1e300;
        int nit = 0;
        i = j = e = f = 0;
        c = s = t = thta = 0.0;
        error = 0;

#define EPS_JACOBI 1e-10

        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                diagM[i][j] = M[i][j];
                v[i][j] = 0.0;
            }
            v[i][i] = 1.0;
        }

        do {
            nit++;
            error = maxMatrix(diagM, &e, &f, &max);
            if (error) {
                printf("No maximum element found.\n");
            }
            if (max > 0) {
                // rotate matrix
                thta = (diagM[f][f] - diagM[e][e]) / (2 * diagM[e][f]);
                if (thta < 0)
                    t = -1. / (cuda::math::abs(thta) + cuda::math::sqrt(thta * thta + 1));
                else
                    t = 1. / (cuda::math::abs(thta) + cuda::math::sqrt(thta * thta + 1));
                // the elements of the rotation matrix
                c = 1. / (cuda::math::sqrt(t * t + 1));
                s = t * c;
                // do diagM' = A^T diagM A
                rotateMatrix(diagM, c, s, e, f);
                identityMatrix(A);
                A[e][e] = c;
                A[f][f] = c;
                A[e][f] = -s;
                A[f][e] = s;
                // calculate the eigenvectors
                multiplyMatrix(v, A, vtmp);
                copyMatrix(vtmp, v);
            }
        } while (max > EPS_JACOBI);

        for (i = 0; i < DIM; i++) {
            eigenvalues[i] = diagM[i][i];
        }
        return nit;
    }


    //
    // computes the eigenvalues of the _symmetric_ matrix M
    // using the jacobi method
    // returns the greatest eigenvalue
    //
    __device__ real calculateMaxEigenvalue(real M[DIM][DIM]) {
        int i, j;
        real diagM[DIM][DIM] = {0.0,};
        real c, s, t, thta;
        int e, f;
        int error;
        real max;
        real max_ev;
        int nit = 0;
        i = j = e = f = 0;
        c = s = t = thta = 0.0;
        max = max_ev = 0;
        error = 0;


#define EPS_JACOBI 1e-10

        for (i = 0; i < DIM; i++)
            for (j = 0; j < DIM; j++)
                diagM[i][j] = M[i][j];

        do {
            nit++;
            error = maxMatrix(diagM, &e, &f, &max);
            if (error) {
                printf("No maximum element found.\n");
            }
            if (max > 0) {
                // rotate matrix
                thta = (diagM[f][f] - diagM[e][e]) / (2 * diagM[e][f]);
                if (thta < 0)
                    t = -1. / (cuda::math::abs(thta) + cuda::math::sqrt(thta * thta + 1));
                else
                    t = 1. / (cuda::math::abs(thta) + cuda::math::sqrt(thta * thta + 1));
                // the elements of the rotation matrix
                c = 1. / (cuda::math::sqrt(t * t + 1));
                s = t * c;
                // do diagM' = A^T diagM A
                rotateMatrix(diagM, c, s, e, f);
            }
        } while (max > EPS_JACOBI || nit < 5);

        max_ev = diagM[0][0];
        for (i = 1; i < DIM; i++) {
            if (diagM[i][i] > max_ev) {
                max_ev = diagM[i][i];
            }
        }
        return max_ev;
    }

    __device__ real det2x2(real a, real b, real c, real d) {
        return a * d - c * b;
    }

    __device__ int invertMatrix(real *m, real *inverted) {
        real det;
#if (DIM == 2)
        real a, b, c, d;
        a = m[0*DIM+0];
        b = m[0*DIM+1];
        c = m[1*DIM+0];
        d = m[1*DIM+1];

        det = det2x2(a,b,c,d);
        //  if (det < 1e-8) return -1;
        // if (det < 1e-10) det = 1e-10;
        det = 1./det;

        inverted[0*DIM+0] = det*d;
        inverted[0*DIM+1] = -det*b;
        inverted[1*DIM+0] = -det*c;
        inverted[1*DIM+1] = det*a;
#elif (DIM == 3)
        det = m[0 * DIM + 0] * (m[1 * DIM + 1] * m[2 * DIM + 2] - m[2 * DIM + 1] * m[1 * DIM + 2])
            - m[0 * DIM + 1] * (m[1 * DIM + 0] * m[2 * DIM + 2] - m[1 * DIM + 2] * m[2 * DIM + 0])
            + m[0 * DIM + 2] * (m[1 * DIM + 0] * m[2 * DIM + 1] - m[1 * DIM + 1] * m[2 * DIM + 0]);

        // inverse determinante

        if (det < FLOAT_ZERO_TOLERANCE){
            printf("ERROR: matrix to be inverted is probably singular: det(M) = %e\n", det);
            return -1;
        }
        det = 1.0 / det;

        inverted[0*DIM+0] = (m[1*DIM+ 1] * m[2*DIM+ 2] - m[2*DIM+ 1] * m[1*DIM+ 2]) * det;
        inverted[0*DIM+1] = (m[0*DIM+ 2] * m[2*DIM+ 1] - m[0*DIM+ 1] * m[2*DIM+ 2]) * det;
        inverted[0*DIM+2] = (m[0*DIM+ 1] * m[1*DIM+ 2] - m[0*DIM+ 2] * m[1*DIM+ 1]) * det;
        inverted[1*DIM+0] = (m[1*DIM+ 2] * m[2*DIM+ 0] - m[1*DIM+ 0] * m[2*DIM+ 2]) * det;
        inverted[1*DIM+1] = (m[0*DIM+ 0] * m[2*DIM+ 2] - m[0*DIM+ 2] * m[2*DIM+ 0]) * det;
        inverted[1*DIM+2] = (m[1*DIM+ 0] * m[0*DIM+ 2] - m[0*DIM+ 0] * m[1*DIM+ 2]) * det;
        inverted[2*DIM+0] = (m[1*DIM+ 0] * m[2*DIM+ 1] - m[2*DIM+ 0] * m[1*DIM+ 1]) * det;
        inverted[2*DIM+1] = (m[2*DIM+ 0] * m[0*DIM+ 1] - m[0*DIM+ 0] * m[2*DIM+ 1]) * det;
        inverted[2*DIM+2] = (m[0*DIM+ 0] * m[1*DIM+ 1] - m[1*DIM+ 0] * m[0*DIM+ 1]) * det;
#endif

        return 1;
    }
}
