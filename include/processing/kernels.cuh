/**
 * @file kernels.cuh
 * @brief ...
 *
 * ...
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_KERNELS_CUH
#define MILUPHPC_KERNELS_CUH

#include "../parameter.h"
#if TARGET_GPU
#include "../particles.cuh"
#include "../materials/material.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"

namespace Processing {

    namespace Kernel {

        /**
         * @brief Particles within radius/radii.
         *
         * > Corresponding wrapper function: ::Processing::Kernel::Launch::particlesWithinRadii()
         *
         * @param particles Particles class instance
         * @param particlesWithin
         * @param deltaRadial
         * @param n
         */
        __global__ void particlesWithinRadii(Particles *particles, int *particlesWithin, real deltaRadial, int n);

        /**
         * @brief Convert cartesian to radial.
         *
         * > Corresponding wrapper function: ::Processing::Kernel::Launch::cartesianToRadial()
         *
         * @tparam T
         * @param particles
         * @param particlesWithin
         * @param input
         * @param output
         * @param deltaRadial
         * @param n
         */
        template<typename T>
        __global__ void
        cartesianToRadial(Particles *particles, int *particlesWithin, T *input, T *output, real deltaRadial, int n);

        namespace Launch {
            /**
             * @brief Wrapper for ::Processing::Kernel::particlesWithinRadii().
             */
            void particlesWithinRadii(Particles *particles, int *particlesWithin, real deltaRadial, int n);

            /**
             * @brief Wrapper for ::Processing::Kernel::cartesianToRadial().
             */
            template<typename T>
            void
            cartesianToRadial(Particles *particles, int *particlesWithin, T *input, T *output, real deltaRadial, int n);
        }
    }
}
#endif
#endif //MILUPHPC_KERNELS_CUH
