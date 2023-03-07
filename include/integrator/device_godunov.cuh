/**
 * @file device_godunov.cuh
 * @brief Device functions and kernels for Godunov's method integrator.
 *
 * @author Johannes S. Martin
 * @bug no known bugs
*/

#ifndef MILUPHPC_DEVICE_GODUNOV_H
#define MILUPHPC_DEVICE_GODUNOV_H

#include "../particles.cuh"
#include <assert.h>

/// device godunov type integrator
namespace GodunovNS {

    /// kernel functions
    namespace Kernel {

        /**
         * @brief particle state update via fluxes and successive position update
         *
         * > Corresponding wrapper function: ::GodunovNS::Kernel::Launch::update()
         *
         * @param particles Particles class instance
         * @param numParticles Number of particles to be advanced
         * @param dt time step
         */
        __global__ void update(Particles *particles, int numParticles, real dt);

        /// wrapped kernel functions
        namespace Launch {
            /**
             * @brief Wrapper for ::GodunovNS::Kernel::update().
             *
             * @return Wall time of execution
             */
            real update(Particles *particles, int numParticles, real dt);
        }
    }

}

#endif //MILUPHPC_DEVICE_GODUNOV_H