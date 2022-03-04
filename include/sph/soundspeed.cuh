/**
 * @file soundspeed.cuh
 * @brief Calculating and initializing the speed of sound in dependence of the used equation of state.
 *
 * This file contains the speed of sound calculation and initialization.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_SOUNDSPEED_CUH
#define MILUPHPC_SOUNDSPEED_CUH

#include "../particles.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "../materials/material.cuh"
#include "kernel.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"

namespace SPH {
    namespace Kernel {
        /**
         * @brief Initialize the speed of sound \f$ c_s \f$.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::initializeSoundSpeed()
         *
         * @note Some materials only initialize the speed of sound and others calculate throughout the simulation.
         *
         * @param particles Particles class instance
         * @param materials Material class instance
         * @param numParticles number of particles
         */
        __global__ void initializeSoundSpeed(Particles *particles, Material *materials, int numParticles);

        /**
         * @brief Calculate the speed of sound \f$ c_s \f$.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::calculateSoundSpeed()
         *
         * @note Some materials only initialize the speed of sound and others calculate throughout the simulation.
         *
         * @param particles Particles class instance
         * @param materials Material class instance
         * @param numParticles number of particles
         */
        __global__ void calculateSoundSpeed(Particles *particles, Material *materials, int numParticles);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::initializeSoundSpeed().
             *
             * @param particles Particles class instance
             * @param materials Material class instance
             * @param numParticles number of particles
             * @return Wall time of kernel execution.
             */
            real initializeSoundSpeed(Particles *particles, Material *materials, int numParticles);

            /**
             * @brief Wrapper for ::SPH::Kernel::calculateSoundSpeed().
             *
             * @param particles Particles class instance
             * @param materials Material class instance
             * @param numParticles number of particles
             * @return Wall time of kernel execution.
             */
            real calculateSoundSpeed(Particles *particles, Material *materials, int numParticles);
        }
    }
}

#endif //MILUPHPC_SOUNDSPEED_CUH
