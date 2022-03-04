/**
 * @file density.cuh
 * @brief SPH density calculation.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_DENSITY_CUH
#define MILUPHPC_DENSITY_CUH

#include "../particles.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "kernel.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"
#include "sph.cuh"

class density {

};

namespace SPH {

    namespace Kernel {

        /**
         * @brief Calculate the density \f$ \rho \f$.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::calculateDensity()
         *
         * \f[
         * The density is given by the kernel sum
         *  \begin{equation}
	            \rho_a = \sum_{b} m_b W_{ab} \, .
         *  \end{equation}
         * \f]
         *
         * @param kernel SPH smoothing kernel
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param interactions interaction list/interaction partners
         * @param numParticles amount of particles
         */
        __global__ void calculateDensity(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::calculateDensity().
             *
             * @param kernel SPH smoothing kernel
             * @param tree Tree class instance
             * @param particles Particles class instance
             * @param interactions interaction list/interaction partners
             * @param numParticles amount of particles
             */
            real calculateDensity(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles);
        }

    }
}


#endif //MILUPHPC_DENSITY_CUH
