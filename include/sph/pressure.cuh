/**
 * @file pressure.cuh
 * @brief Pressure calculation in dependence of the equation of states.
 *
 * This file contains the pressure calculation for different equation of states
 * and a CUDA kernel for executing the correct device function for each particle.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_PRESSURE_CUH
#define MILUPHPC_PRESSURE_CUH

#include "../particles.cuh"
#include "../parameter.h"
#include "../materials/material.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"

class pressure {

};

/// Equation of states
namespace EOS {
    /**
     * @brief Polytropic gas.
     *
     * Refer to ::EquationOfStates.
     *
     * @param materials Material class instance
     * @param particles Particles class instance
     * @param index Relevant particle index
     */
    __device__ void polytropicGas(Material *materials, Particles *particles, int index);

    /**
     * @brief Isothermal gas.
     *
     * Refer to ::EquationOfStates.
     *
     * @param materials Material class instance
     * @param particles Particles class instance
     * @param index Relevant particle index
     */
    __device__ void isothermalGas(Material *materials, Particles *particles, int index);

    /**
     * @brief Ideal gas.
     *
     * Refer to ::EquationOfStates.
     *
     * @param materials Material class instance
     * @param particles Particles class instance
     * @param index Relevant particle index
     */
    __device__ void idealGas(Material *materials, Particles *particles, int index);
}

namespace SPH {
    namespace Kernel {
        /**
         * @brief Calculate the pressure.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::calculatePressure()
         *
         * @param materials Material class instance
         * @param particles Particles class instance
         * @param numParticles amount of particles
         */
        __global__ void calculatePressure(Material *materials, Particles *particles, int numParticles);

        namespace Launch {
            /**
             * @brief Wrapper for ::SPH::Kernel::calculatePressure().
             *
             * @param materials Material class instance
             * @param particles Particles class instance
             * @param numParticles amount of particles
             * @return Wall time for kernel execution
             */
            real calculatePressure(Material *materials, Particles *particles, int numParticles);
        }
    }

}




#endif //MILUPHPC_PRESSURE_CUH
