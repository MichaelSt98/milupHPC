/**
 * @file volume_partition.cuh
 * @brief MFV/MFM density calculation.
 *
 * @author Johannes S. Martin
 * @bug no known bugs
 */

#ifndef MILUPHPC_VOLUME_PARTITION_CUH
#define MILUPHPC_VOLUME_PARTITION_CUH

#include "../sph/kernel.cuh"
#include "../particles.cuh"
#include "../cuda_utils/linalg.cuh"

namespace MFV {

    namespace Compute {

        /**
         * @brief derivative by kernel size/smoothing length of cubbic spline kernel
         *
         * @param[out] dWdh derivative of kernel by smoothing length
         * @param[in] dx vector between particles i and j
         * @param[in] sml smoothing length or kernel size
         *
         */
        __device__ void dWdh_cubicSpline(real &dWdh, real dx[DIM], real sml);

        /**
         * @brief worker function to compute invere volume \f$ \omega(\vec{x}_i) \f$.
         *
         * @param[out] omg inverse volume/number density
         * @param[in] i index of particle \f$ i \f$
         * @param[in] kernel smoothing kernel
         * @param[in] particles Particles class instance
         * @param[in] interactions interaction list/interaction partners
         * @returs derivative of kernel by smoothing length for variable smoothing length mode
         */
        __device__ real inverseVolume(real &omg, int i, ::SPH::SPH_kernel kernel, Particles *particles, int *interactions);
    }

    namespace Kernel {

        /**
         * @brief Calculate the density \f$ \rho \f$.
         *
         * > Corresponding wrapper function: ::MFV::Kernel::Launch::calculateDensity()
         *
         * In order to compute the density, the simulation domain is partitioned into effective volumes of particles.
         * With this volume partition the density can be calculated
         *
         * \f[
         * The volume partition is given by the sum over all neighboring particles $j$
         *  \begin{equation}
	            \omega\left(\vec{x}\right) = \sum_j W\left(\vec{x}-\vec{x}_j, h\left(\vec{x}\right)\right)
         *  \end{equation}
         *  and the density of particle $i$ is calculated by
         *  \begin{equation}
         *      \rho_i = m_i \omega(\vec{x}_i)
         *  \end{equation}
         * \f]
         *
         * @param kernel smoothing kernel
         * @param particles Particles class instance
         * @param interactions interaction list/interaction partners
         * @param numParticles amount of particles
         */
        __global__ void calculateDensity(::SPH::SPH_kernel kernel, Particles *particles,
                                         int *interactions, int numParticles);

        /**
         * @brief Computing vector weights needed for second-order accurate gradients and effective volume computation
         *
         * * > Corresponding wrapper function: ::MFV::Kernel::Launch::computeVectorWeights()
         *
         * @param kernel smoothing kernel
         * @param particles Particles class instance
         * @param interactions interaction list/interaction partners
         * @param numParticles amount of particles
         * @param critCondNum critical condition number for matrix E_i
         */
        __global__ void computeVectorWeights(::SPH::SPH_kernel kernel, Particles *particles,
                                             int *interactions, int numParticles, real *critCondNum);


        namespace Launch {
            /**
             * @brief Wrapper for ::MFV::Kernel::calculateDensity().
             *
             * @param kernel smoothing kernel
             * @param particles Particles class instance
             * @param interactions interaction list/interaction partners
             * @param numParticles amount of particles
             */
            real calculateDensity(::SPH::SPH_kernel kernel, Particles *particles,
                                  int *interactions, int numParticles);

            /**
             * @brief Wrapper for ::MFV::Kernel::computeVectorWeights().
             *
             * @param kernel smoothing kernel
             * @param particles Particles class instance
             * @param interactions interaction list/interaction partners
             * @param numParticles amount of particles
             * @param critCondNum critical condition number for matrix E_i
             */
            real computeVectorWeights(::SPH::SPH_kernel kernel, Particles *particles,
                                  int *interactions, int numParticles, real *critCondNum);
        }

    }

}

#endif //MILUPHPC_VOLUME_PARTITION_CUH