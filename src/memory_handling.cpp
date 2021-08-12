//
// Created by Michael Staneker on 12.08.21.
//

#include "../include/memory_handling.h"
/*
memory_handling::memory_handling() {

}

memory_handling::memory_handling(integer particleCount) : particleCount(particleCount) {

}

void memory_handling::allocateParticles() {

    h_particles = new Particles();
    h_mass = new real[particleCount];
    h_x = new real[particleCount];
    h_y = new real[particleCount];
    h_z = new real[particleCount];
    h_vx = new real[particleCount];
    h_vy = new real[particleCount];
    h_vz = new real[particleCount];
    h_ax = new real[particleCount];
    h_ay = new real[particleCount];
    h_az = new real[particleCount];

    h_particles->setParticle(particleCount, h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az);

    gpuErrorcheck(cudaMalloc((void**)&d_mass, particleCount * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_x, particleCount * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_y, particleCount * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_z, particleCount * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_vx, particleCount * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_vy, particleCount * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_vz, particleCount * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_ax, particleCount * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_ay, particleCount * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_az, particleCount * sizeof(real)));

    gpuErrorcheck(cudaMalloc((void**)&d_particles, sizeof(Particles)));
    ParticlesNS::launchSetKernel(d_particles, particleCount, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);
    gpuErrorcheck( cudaPeekAtLastError() );
    gpuErrorcheck( cudaDeviceSynchronize() );
}

void memory_handling::getParticlesObjects(Particles *host_particles, Particles *device_particles) {
    host_particles = h_particles;
    device_particles = d_particles;
}

memory_handling::~memory_handling() {

    delete h_particles;
    delete [] h_mass;
    delete [] h_x;
    delete [] h_y;
    delete [] h_z;
    delete [] h_vx;
    delete [] h_vy;
    delete [] h_vz;
    delete [] h_ax;
    delete [] h_ay;
    delete [] h_az;
    // device particle entries
    gpuErrorcheck(cudaFree(d_particles));
    gpuErrorcheck(cudaFree(d_mass));
    gpuErrorcheck(cudaFree(d_x));
    gpuErrorcheck(cudaFree(d_y));
    gpuErrorcheck(cudaFree(d_z));
    gpuErrorcheck(cudaFree(d_vx));
    gpuErrorcheck(cudaFree(d_vy));
    gpuErrorcheck(cudaFree(d_vz));
    gpuErrorcheck(cudaFree(d_ax));
    gpuErrorcheck(cudaFree(d_ay));
    gpuErrorcheck(cudaFree(d_az));

}*/
