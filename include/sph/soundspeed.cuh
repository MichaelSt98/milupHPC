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

        __global__ void initializeSoundSpeed(Particles *particles, Material *materials, int numParticles);

        __global__ void calculateSoundSpeed(Particles *particles, Material *materials, int numParticles);

        namespace Launch {
            real initializeSoundSpeed(Particles *particles, Material *materials, int numParticles);
            real calculateSoundSpeed(Particles *particles, Material *materials, int numParticles);
        }
    }
}

#endif //MILUPHPC_SOUNDSPEED_CUH
