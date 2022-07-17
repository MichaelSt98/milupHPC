/**
 * @file viscosity.cuh
 * @brief Stress.
 *
 * More detailed information.
 *
 * @author Michael Staneker
 * @bug no known bugs
 * @todo implement
 */
#ifndef MILUPHPC_VISCOSITY_CUH
#define MILUPHPC_VISCOSITY_CUH

#include "../parameter.h"
#if TARGET_GPU
#include "../particles.cuh"
#include "../materials/material.cuh"
#include "kernel.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"

namespace SPH {
    namespace Kernel {

// TODO: functions only needed for NAVIER_STOKES
#if NAVIER_STOKES
        __global__ void calculate_shear_stress_tensor(::SPH::SPH_kernel kernel, Material *materials, Particles *particles, int *interactions, int numRealParticles);
#endif

#if NAVIER_STOKES
        __global__ void calculate_kinematic_viscosity(::SPH::SPH_kernel kernel, Material *materials, Particles *particles, int *interactions, int numRealParticles);
#endif
    }
}

#endif // TARGET_GPU
#endif //MILUPHPC_VISCOSITY_CUH
