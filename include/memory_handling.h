//
// Created by Michael Staneker on 12.08.21.
//

#ifndef MILUPHPC_MEMORY_HANDLING_H
#define MILUPHPC_MEMORY_HANDLING_H

#include "particles.cuh"
#include "cuda_utils/cuda_utilities.cuh"

/*class memory_handling {

public:

    Particles *h_particles;
    Particles *d_particles;
    // host particle entries
    real *h_mass;
    real *h_x, *h_y, *h_z;
    real *h_vx, *h_vy, *h_vz;
    real *h_ax, *h_ay, *h_az;
    // device particle entries
    real *d_mass;
    real *d_x, *d_y, *d_z;
    real *d_vx, *d_vy, *d_vz;
    real *d_ax, *d_ay, *d_az;

public:

    integer particleCount;

    memory_handling();
    memory_handling(integer particleCount);
    ~memory_handling();

    void allocateParticles();

    void getParticlesObjects(Particles *host_particles, Particles *device_particles);

};*/


#endif //MILUPHPC_MEMORY_HANDLING_H
