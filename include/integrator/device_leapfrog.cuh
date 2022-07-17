/**
 * @file device_leapfrog.cuh
 * @brief Device functions and kernels for the leapfrog integrator.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_DEVICE_LEAPFROG_CUH
#define MILUPHPC_DEVICE_LEAPFROG_CUH

#include "../particles.cuh"
#if TARGET_GPU
#include <assert.h>

/// leapfrog integrator
namespace LeapfrogNS {

    /// kernel functions
    namespace Kernel {

        /**
         * @brief Update/move/advance particles.
         *
         * > Corresponding wrapper function: ::LeapfrogNS::Kernel::Launch::update()
         *
         * @param particles Particles class instance
         * @param n Number of particles to be advanced
         * @param dt time step
         */
        __global__ void updateX(Particles *particles, integer n, real dt);

        __global__ void updateV(Particles *particles, integer n, real dt);

        /// wrapped kernel functions
        namespace Launch {
            /**
             * @brief Wrapper for ::LeapfrogNS::Kernel::update().
             *
             * @return Wall time of execution
             */
            real updateX(Particles *particles, integer n, real dt);

            real updateV(Particles *particles, integer n, real dt);
        }
    }

}

#endif // TARGET_GPU
#endif //MILUPHPC_DEVICE_LEAPFROG_CUH