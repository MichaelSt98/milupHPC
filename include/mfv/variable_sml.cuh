/**
 * @file variable_sml.cuh
 * @brief MFV/MFM variable smoothing length neighbor search.
 *
 * @author Johannes S. Martin
 * @bug no known bugs
 */

#ifndef MILUPHPC_VARIABLE_SML_CUH
#define MILUPHPC_VARIABLE_SML_CUH

#include "../particles.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "../helper.cuh"
#include "../materials/material.cuh"
#include "volume_partition.cuh"

namespace MFV {

    namespace Kernel {

        /**
         * @brief Guessing smoothing length according to the
         *
         * @param particles particles class instance
         * @param numParticles number of particles
         * @param dt current timestep
         */
        __global__ void guessSML(Particles *particles, int numParticles, real dt);

        /**
         * @brief Fixed-radius near neighbor search (nested stack method).
         *
         * > Corresponding wrapper function: ::MFV::Kernel::Launch::variableSML_FRNN()
         *
         * This function utilizes the algorithm from ::SPH::Kernel::fixedRadiusNN() and redos the neighbor
         * search until the following criterion is fulfilled:
         *
         * \f[
         *   \begin{equation}
         *     N_\text{NN} = C_d h_i^d \sum_j W(\vec{x}_j - \vec{x}_i, h_i) \ , \, C_1=1, C_2 = \pi, C_3 = \frac{4\pi}{3}
         *   \end{equation}
         * \f]
         *
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[out] interactions interaction partners
         * @param[in] radius maximum distance of particles in one dimension // TODO: check why this is necessary
         * @param[in] numParticlesLocal number of local particles
         * @param[in] numParticles number of particles in total
         * @param[in] numNodes number of nodes
         * @param[in] materials Material class instance
         * @param[in] kernel smoothing kernel
         */
        __global__ void variableSML_FRNN(Tree *tree, Particles *particles, integer *interactions, real radius,
                                         integer numParticlesLocal, integer numParticles, integer numNodes,
                                         Material *materials, ::SPH::SPH_kernel kernel);

        namespace Launch {

            /**
             * @brief Wrapper for ::MFV::Kernel::guessSML().
             *
             * @return Wall time for kernel execution
             */
            real guessSML(Particles *particles, int numParticles, real dt);

            /**
             * @brief Wrapper for ::MFV::Kernel::variableSML_FRNN().
             *
             * @return Wall time for kernel execution
             */
            real variableSML_FRNN(Tree *tree, Particles *particles, integer *interactions, real radius,
                                  integer numParticlesLocal, integer numParticles, integer numNodes,
                                  Material *materials, ::SPH::SPH_kernel kernel);


        }
    }

}

#endif // MILUPHPC_VARIABLE_SML_CUH