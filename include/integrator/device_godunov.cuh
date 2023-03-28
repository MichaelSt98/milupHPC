/**
 * @file device_godunov.cuh
 * @brief Device functions and kernels for Godunov's method integrator.
 *
 * @author Johannes S. Martin
 * @bug no known bugs
*/

#ifndef MILUPHPC_DEVICE_GODUNOV_H
#define MILUPHPC_DEVICE_GODUNOV_H

#include "../parameter.h"
#include "../particles.cuh"
#include "../simulation_time.cuh"
#include <assert.h>

#define ENERGY_FLOOR 1e-11

/// device godunov type integrator
namespace GodunovNS {

    /// kernel functions
    namespace Kernel {

        /**
         *
         * Selecting timestep for MFV/MFM schemes depending on the signal velocity and kernel size
         * all interacting particles
         *
         * @param simulationTime simulation time instance
         * @param particles particles instance
         * @param numParticles number of particles on current process
         * @param dtBlockShared container for findin minimum timestep amongst all particles
         * @param blockCount counter variable for compute blocks
         */
        __global__ void selectTimestep(SimulationTime *simulationTime, Particles *particles, int numParticles,
                                       real *dtBlockShared, int *blockCount);

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

            /**
             *
             * @brief Wrapper for ::GodunovNS::Kernel::selectTimestep().
             *
             * @param multiProcessorCount
             * @param simulationTime
             * @param particles
             * @param numParticles
             * @param dtBlockShared
             * @param blockCount
             * @return Wall time of execution
             */
            real selectTimestep(int multiProcessorCount, SimulationTime *simulationTime, Particles *particles,
                                int numParticles, real *dtBlockShared, int *blockCount);

        }
    }

}

#endif //MILUPHPC_DEVICE_GODUNOV_H