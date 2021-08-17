//
// Created by Michael Staneker on 12.08.21.
//

#ifndef MILUPHPC_PARTICLE_HANDLER_H
#define MILUPHPC_PARTICLE_HANDLER_H

#include "parameter.h"
#include "particles.cuh"
#include "utils/logger.h"

class ParticleHandler {

public:

    integer numParticles;
    integer numNodes;

    real *h_mass;
    real *h_x, *h_vx, *h_ax;
#if DIM > 1
    real *h_y, *h_vy, *h_ay;
#if DIM == 3
    real *h_z, *h_vz, *h_az;
#endif
#endif

    // device particle entries
    real *d_mass;
    real *d_x, *d_vx, *d_ax;
#if DIM > 1
    real *d_y, *d_vy, *d_ay;
#if DIM == 3
    real *d_z, *d_vz, *d_az;
#endif
#endif

    //Particles h_particles;
    //Particles d_particles;
    Particles *h_particles;
    Particles *d_particles;

    ParticleHandler(integer numParticles, integer numNodes);
    ~ParticleHandler();

    void positionToDevice();
    void velocityToDevice();
    void accelerationToDevice();
    void distributionToDevice(bool velocity=true, bool acceleration=true);
    void positionToHost();
    void velocityToHost();
    void accelerationToHost();
    void distributionToHost(bool velocity=true, bool acceleration=true);

};


#endif //MILUPHPC_PARTICLE_HANDLER_H
