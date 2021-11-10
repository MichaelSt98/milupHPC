#ifndef MILUPHPC_DENSITY_CUH
#define MILUPHPC_DENSITY_CUH

#include "../particles.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "kernel.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "cuda_utils/cuda_runtime.h"
#include "sph.cuh"

class density {

};

namespace SPH {

    namespace Kernel {

        __global__ void calculateDensity(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles);

        namespace Launch {
            real calculateDensity(::SPH::SPH_kernel kernel, Tree *tree, Particles *particles, int *interactions, int numParticles);
        }

    }
}


#endif //MILUPHPC_DENSITY_CUH
