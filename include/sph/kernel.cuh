/**
 * @file internal_forces.cuh
 * @brief SPH internal forces.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_KERNEL_CUH
#define MILUPHPC_KERNEL_CUH

#if TARGET_GPU
#include "../parameter.h"
#include "../particles.cuh"
#include "../helper.cuh"

//#include <boost/mpi.hpp>
#include <assert.h>
#include "../parameter.h"
#include "../cuda_utils/linalg.cuh"

namespace SPH {


    /**
     * @brief Function pointer to generic SPH kernel function.
     */
    typedef void (*SPH_kernel)(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real h);

    namespace SmoothingKernel {

        /**
         * @brief Spiky kernel (Desbrun & Cani).
         *
         * With the normalization constant \f$ \sigma \f$ and dimensionality \f$ d \f$ the kernel can be defined as:
         * \f[
         *   \begin{equation}
	            W(q) = \frac{\sigma}{h^d}
	            \begin{cases}
		            (1-q)^3 & \text{for } 0 \leq q < 1 \\
		            0 & \text{for } q \geq 1 \\
	            \end{cases}
	            \quad
	            \sigma =
	            \begin{cases}
		            \frac{10}{\pi} & \text{for } d=2 \\
		            \frac{15}{\pi} & \text{for } d=3 \\
	                \end{cases} \, .
                \end{equation}
         * \f]
         *
         * @param[out] W smoothed value/contribution
         * @param[out] dWdx spatial derivative for each coordinate axis (dimensionality: DIM)
         * @param[out] dWdr spatial derivative
         * @param[in] dx spatial separation (dimensionality: DIM)
         * @param[in] sml smoothing length
         */
        __device__ void spiky(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);

        /**
         * @brief Cubic spline kernel (Monaghan & Lattanzio 1985).
         *
         * With the normalization constant \f$ \sigma \f$ and dimensionality \f$ d \f$ the kernel can be defined as:
         * \f[
         *     \begin{equation}
	                W(q) = \frac{\sigma}{h^d}
	                \begin{cases}
		                (6q^3-6q2+1) &  \text{ for } 0 \leq q < \frac{1}{2} \\
		                2(1-q)^3 &  \text{ for } \frac{1}{2} \leq q \leq 1 \\
		                0 &  \text{ for } q > 1 \\
	                \end{cases}
	                \quad
	                \sigma =
	                \begin{cases}
		                \frac{4}{3} & \text{for } d = 1 \\
		                \frac{40}{7 \pi} & \text{for } d = 2\\
		                \frac{8}{\pi} & \text{for } d = 3 \\
	                \end{cases} \, .
                \end{equation}
         * \f]
         *
         * @param[out] W smoothed value/contribution
         * @param[out] dWdx spatial derivative for each coordinate axis (dimensionality: DIM)
         * @param[out] dWdr spatial derivative
         * @param[in] dx spatial separation (dimensionality: DIM)
         * @param[in] sml smoothing length
         */
        __device__ void cubicSpline(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);

        /**
         * Wendland C2 (Dehnen & Aly, 2012)
         *
         * With the normalization constant \f$ \sigma \f$ and dimensionality \f$ d \f$ the kernel can be defined as:
         * \f[
         *      \begin{equation}
	                W (q) = \frac{\sigma}{h^d}
	                \begin{cases}
		                (1-q)^4 (1+4q) & \text{for }d=2\text{, } d=3 \text{ and } 0 \leq q < 1 \\
		                (1-q)^3 (1+3q) & \text{for }d=1 \text{ and } 0 \leq q < 1 \\
		                0 & q \geq 1
	                \end{cases}
	                \quad
	                \sigma =
	                \begin{cases}
		                \frac{5}{4} & \text{for } d=1 \\
		                \frac{7}{\pi} & \text{for } d=2 \\
		                \frac{21}{2 \pi} & \text{for }d=3 \\
	                \end{cases} \, .
                \end{equation}
         * \f]
         *
         * @param[out] W smoothed value/contribution
         * @param[out] dWdx spatial derivative for each coordinate axis (dimensionality: DIM)
         * @param[out] dWdr spatial derivative
         * @param[in] dx spatial separation (dimensionality: DIM)
         * @param[in] sml smoothing length
         */
        __device__ void wendlandc2(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);

        /**
         * Wendland C4 (Dehnen & Aly, 2012)
         *
         * With the normalization constant \f$ \sigma \f$ and dimensionality \f$ d \f$ the kernel can be defined as:
         * \f[
         *      \begin{equation}
	                W (q) = \frac{\sigma}{h^d}
	                \begin{cases}
		                (1-q)^6 (1 + 6q + \frac{35}{3}q^2) & \text{for } d=2\text{, } d=3 \text{ and } 0 \leq q < 1 \\
		                (1-q)^5 (1+5q+8q^2) & \text{for } d=1 \text{ and } 0 \leq q < 1 \\
		                0 & q \geq 1
	                \end{cases}
	                \quad
	                \sigma =
	                \begin{cases}
		                \frac{3}{2} & \text{for } d=1 \\
		                \frac{9}{\pi} & \text{for } d=2 \\
		                \frac{495}{32 \pi} & \text{for } d=3 \\
	                \end{cases}
                \end{equation}
         * \f]
         *
         * @param[out] W smoothed value/contribution
         * @param[out] dWdx spatial derivative for each coordinate axis (dimensionality: DIM)
         * @param[out] dWdr spatial derivative
         * @param[in] dx spatial separation (dimensionality: DIM)
         * @param[in] sml smoothing length
         */
        __device__ void wendlandc4(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);

        /**
         * Wendland C6 (Dehnen & Aly, 2012)
         *
         * With the normalization constant \f$ \sigma \f$ and dimensionality \f$ d \f$ the kernel can be defined as:
         * \f[
         *      \begin{equation}
	                W (q) = \frac{\sigma}{h^d}
	                \begin{cases}
		                (1-q)^8 (1 + 8q + 25 q^2 + 32 q^3) & \text{for } d=2\text{, } d=3 \text{ and } 0 \leq q < 1 \\
		                (1-q)^7 (1 + 7q + 19 q^2 + 21 q^3) & \text{for } d=1 \text{ and } 0 \leq q < 1 \\
		                0 & q \geq 1
	                \end{cases}
	                \quad
	                \sigma =
	                \begin{cases}
		                \frac{55}{32} & \text{for } d=1 \\
		                \frac{78}{7 \pi} & \text{for } d=2 \\
		                \frac{1365}{64 \pi} & \text{for } d=3 \\
	                \end{cases} \, .
                \end{equation}
         * \f]
         *
         * @param[out] W smoothed value/contribution
         * @param[out] dWdx spatial derivative for each coordinate axis (dimensionality: DIM)
         * @param[out] dWdr spatial derivative
         * @param[in] dx spatial separation (dimensionality: DIM)
         * @param[in] sml smoothing length
         */
        __device__ void wendlandc6(real *W, real dWdx[DIM], real *dWdr, real dx[DIM], real sml);
    }


    //TODO: implement:
    /**
     * @brief Calculates the kernel for the tensile instability fix (Monaghan 2000).
     *
     * @todo implement `fixTensileInstability()`.
     *
     * @param kernel
     * @param particles
     * @param p1
     * @param p2
     * @return
     */
    CUDA_CALLABLE_MEMBER real fixTensileInstability(SPH_kernel kernel, Particles *particles, int p1, int p2);

#if (NAVIER_STOKES || BALSARA_SWITCH || INVISCID_SPH || INTEGRATE_ENERGY)

    /**
     * @brief Calculates \f$ \nabla \cdot \vec{v} \f$ and \f$ \nabla \times \vec{v} \f$.
     */
    namespace Kernel {
        __global__ void CalcDivvandCurlv(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles);
        namespace Launch {
            real CalcDivvandCurlv(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles);
        }
    }
#endif

#if ZERO_CONSISTENCY //SHEPARD_CORRECTION

    /**
     * @brief Calculates the zeroth order corrections for the kernel sum.
     */
    __global__ void shepardCorrection(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles);
#endif

#if LINEAR_CONSISTENCY //TENSORIAL_CORRECTION

    /**
     * @brief Calculates the tensorial correction factors for linear consistency.
     */
    __global__ void tensorialCorrection(SPH_kernel kernel, Particles *particles, int *interactions, int numParticles);
#endif

}

#endif // TARGET_GPU
#endif //MILUPHPC_KERNEL_CUH
