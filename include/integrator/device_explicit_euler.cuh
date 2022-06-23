/**
 * @file device_explicit_euler.cuh
 * @brief Device functions and kernels for the explicit euler integrator.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_DEVICE_EXPLICIT_EULER_CUH
#define MILUPHPC_DEVICE_EXPLICIT_EULER_CUH

#include "../particles.cuh"
#include <assert.h>

/// explicit euler integrator
namespace ExplicitEulerNS {

    /// kernel functions
    namespace Kernel {

        /**
         * @brief Update/move/advance particles.
         *
         * > Corresponding wrapper function: ::ExplicitEulerNS::Kernel::Launch::update()
         *
         * @param particles Particles class instance
         * @param n Number of particles to be advanced
         * @param dt time step
         */
        __global__ void update(Particles *particles, integer n, real dt);

        /// wrapped kernel functions
        namespace Launch {
            /**
             * @brief Wrapper for ::ExplicitEulerNS::Kernel::update().
             *
             * @return Wall time of execution
             */
            real update(Particles *particles, integer n, real dt);
        }
    }

}

#endif //MILUPHPC_DEVICE_EXPLICIT_EULER_CUH