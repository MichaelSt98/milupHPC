#ifndef MILUPHPC_KERNELS_CUH
#define MILUPHPC_KERNELS_CUH

#include "../particles.cuh"
#include "../parameter.h"
#include "../materials/material.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"

namespace Processing {

    namespace Kernel {
        __global__ void particlesWithinRadii(Particles *particles, int *particlesWithin, real deltaRadial, int n);

        template<typename T>
        __global__ void
        cartesianToRadial(Particles *particles, int *particlesWithin, T *input, T *output, real deltaRadial, int n);

        namespace Launch {
            void particlesWithinRadii(Particles *particles, int *particlesWithin, real deltaRadial, int n);

            template<typename T>
            void
            cartesianToRadial(Particles *particles, int *particlesWithin, T *input, T *output, real deltaRadial, int n);
        }
    }

}

#endif //MILUPHPC_KERNELS_CUH
