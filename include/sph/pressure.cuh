#ifndef MILUPHPC_PRESSURE_CUH
#define MILUPHPC_PRESSURE_CUH

#include "../particles.cuh"
#include "../parameter.h"
#include "../materials/material.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"

class pressure {

};

namespace EOS {
    __device__ void polytropicGas(Material *materials, Particles *particles, int index);

    __device__ void isothermalGas(Material *materials, Particles *particles, int index);

    __device__ void idealGas(Material *materials, Particles *particles, int index);
}

namespace SPH {
    namespace Kernel {
        __global__ void calculatePressure(Material *materials, Particles *particles, int numParticles);

        namespace Launch {
            real calculatePressure(Material *materials, Particles *particles, int numParticles);
        }
    }

}




#endif //MILUPHPC_PRESSURE_CUH
