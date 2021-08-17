//
// Created by Michael Staneker on 12.08.21.
//

#ifndef MILUPHPC_PARTICLES_CUH
#define MILUPHPC_PARTICLES_CUH

#include "cuda_utils/cuda_utilities.cuh"
//#include "../cuda_utils/cuda_launcher.cuh"
#include "parameter.h"

class Particles {

public:

    integer *numParticles;
    integer *numNodes;

    real *mass;
    real *x, *vx, *ax;
#if DIM > 1
    real *y, *vy, *ay;
#if DIM == 3
    real *z, *vz, *az;
#endif
#endif

    CUDA_CALLABLE_MEMBER Particles();

    //TODO: wouldn't be necessary but better for compilation?
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass,
                                          real *x, real *vx, real *ax);
#if DIM > 1
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx, real *vy, real *ax, real *ay);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx, real *vy,
                                          real *ax, real *ay);
#if DIM == 3
    CUDA_CALLABLE_MEMBER Particles(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z, real *vx,
                                   real *vy, real *vz, real *ax, real *ay, real *az);

    CUDA_CALLABLE_MEMBER void set(integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z, real *vx,
                                          real *vy, real *vz, real *ax, real *ay, real *az);
#endif
#endif

    CUDA_CALLABLE_MEMBER void reset(integer index);

    CUDA_CALLABLE_MEMBER ~Particles();

};

namespace ParticlesNS {

    __global__ void setKernel(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax);

    void launchSetKernel(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *vx, real *ax);
#if DIM > 1
    __global__ void setKernel(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx,
                              real *vy, real *ax, real *ay);

    void launchSetKernel(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *vx,
                         real *vy, real *ax, real *ay);
#if DIM == 3
    __global__ void setKernel(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z, real *vx,
                              real *vy, real *vz, real *ax, real *ay, real *az);

    void launchSetKernel(Particles *particles, integer *numParticles, integer *numNodes, real *mass, real *x, real *y, real *z, real *vx,
                         real *vy, real *vz, real *ax, real *ay, real *az);
#endif
#endif

    __global__ void testKernel(Particles *particles);

    void launchTestKernel(Particles *particles);

}

#endif //MILUPHPC_PARTICLES_CUH
