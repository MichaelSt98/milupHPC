#ifndef MILUPHPC_INTERNAL_FORCES_CUH
#define MILUPHPC_INTERNAL_FORCES_CUH

#include "../subdomain_key_tree/subdomain.cuh"
#include "../materials/material.cuh"
#include "kernel.cuh"

namespace SPH {
    namespace Kernel {
        __global__ void internalForces(::SPH::SPH_kernel kernel, Material *materials, Tree *tree, Particles *particles,
                                       int *interactions, int numRealParticles);

        namespace Launch {
            real internalForces(::SPH::SPH_kernel kernel, Material *materials, Tree *tree, Particles *particles,
                                int *interactions, int numRealParticles);
        }
    }
}

#endif //MILUPHPC_INTERNAL_FORCES_CUH
