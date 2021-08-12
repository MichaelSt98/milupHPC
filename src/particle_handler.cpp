//
// Created by Michael Staneker on 12.08.21.
//

#include "../include/particle_handler.h"

ParticleHandler::ParticleHandler(integer numParticles, integer numNodes) : numParticles(numParticles),
                                                                            numNodes(numNodes) {

    h_mass = new real[numNodes];
    h_x = new real[numNodes];
    h_vx = new real[numNodes];
    h_ax = new real[numNodes];
#if DIM > 1
    h_y = new real[numNodes];
    h_vy = new real[numNodes];
    h_ay = new real[numNodes];
#if DIM == 3
    h_z = new real[numNodes];
    h_vz = new real[numNodes];
    h_az = new real[numNodes];
#endif
#endif
    h_particles = new Particles();

    gpuErrorcheck(cudaMalloc((void**)&d_mass, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_x, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_vx, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_ax, numNodes * sizeof(real)));
#if DIM > 1
    gpuErrorcheck(cudaMalloc((void**)&d_y, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_vy, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_ay, numNodes * sizeof(real)));
#if DIM == 3
    gpuErrorcheck(cudaMalloc((void**)&d_z, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_vz, numNodes * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_az, numNodes * sizeof(real)));
#endif
#endif
    gpuErrorcheck(cudaMalloc((void**)&d_particles, sizeof(Particles)));

#if DIM == 1
    h_particles->setParticle(numParticles, numNodes, h_mass, h_x, h_vx, h_ax);
    ParticlesNS::launchSetKernel(d_particles, numParticles, numNodes, h_mass, h_x, h_vx, h_ax);
#elif DIM == 2
    h_particles->setParticle(numParticles, numNodes, h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay);
    ParticlesNS::launchSetKernel(d_particles, numParticles, numNodes, h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay);
#else
    h_particles->setParticle(numParticles, numNodes, h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az);
    ParticlesNS::launchSetKernel(d_particles, numParticles, numNodes, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);
#endif
    gpuErrorcheck( cudaPeekAtLastError() ); // check CUDA kernel
    gpuErrorcheck( cudaDeviceSynchronize() ); // check CUDA kernel

}

ParticleHandler::~ParticleHandler() {

    delete h_particles;
    delete [] h_mass;
    delete [] h_x;
    delete [] h_vx;
    delete [] h_ax;
#if DIM > 1
    delete [] h_y;
    delete [] h_vy;
    delete [] h_ay;
#if DIM == 3
    delete [] h_z;
    delete [] h_vz;
    delete [] h_az;
#endif
#endif

    // device particle entries
    gpuErrorcheck(cudaFree(d_particles));
    gpuErrorcheck(cudaFree(d_mass));
    gpuErrorcheck(cudaFree(d_x));
    gpuErrorcheck(cudaFree(d_vx));
    gpuErrorcheck(cudaFree(d_ax));
#if DIM > 1
    gpuErrorcheck(cudaFree(d_y));
    gpuErrorcheck(cudaFree(d_vy));
    gpuErrorcheck(cudaFree(d_ay));
#if DIM == 3
    gpuErrorcheck(cudaFree(d_z));
    gpuErrorcheck(cudaFree(d_vz));
    gpuErrorcheck(cudaFree(d_az));
#endif
#endif

}