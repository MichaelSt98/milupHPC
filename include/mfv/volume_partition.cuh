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
#include "../subdomain_key_tree/subdomain.cuh"
#include "../cuda_utils/linalg.cuh"

namespace MFV {

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
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param interactions interaction list/interaction partners
         * @param numParticles amount of particles
         */
        __global__ void calculateDensity(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles,
                                         int *interactions, int numParticles);

        /**
         * @brief Computing vector weights needed for second-order accurate gradients and effective volume computation
         *
         * * > Corresponding wrapper function: ::MFV::Kernel::Launch::computeVectorWeights()
         *
         * @param kernel smoothing kernel
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param interactions interaction list/interaction partners
         * @param numParticles amount of particles
         */
        __global__ void computeVectorWeights(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles,
                                             int *interactions, int numParticles);


        namespace Launch {
            /**
             * @brief Wrapper for ::MFV::Kernel::calculateDensity().
             *
             * @param kernel smoothing kernel
             * @param tree Tree class instance
             * @param particles Particles class instance
             * @param interactions interaction list/interaction partners
             * @param numParticles amount of particles
             */
            real calculateDensity(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles,
                                  int *interactions, int numParticles);

            /**
             * @brief Wrapper for ::MFV::Kernel::computeVectorWeights().
             *
             * @param kernel smoothing kernel
             * @param tree Tree class instance
             * @param particles Particles class instance
             * @param interactions interaction list/interaction partners
             * @param numParticles amount of particles
             */
            real computeVectorWeights(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles,
                                  int *interactions, int numParticles);
        }

    }

}

#endif //MILUPHPC_VOLUME_PARTITION_CUH