#include "../include/particle_handler.h"

ParticleHandler::ParticleHandler(integer numParticles, integer numNodes) : numParticles(numParticles),
                                                                            numNodes(numNodes) {

    Logger(INFO) << "numParticles: " << numParticles << "   numNodes: " << numNodes;

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
    h_particles->set(numParticles, numNodes, h_mass, h_x, h_vx, h_ax);
    ParticlesNS::Kernel::Launch::set(d_particles, numParticles, numNodes, h_mass, h_x, h_vx, h_ax);
#elif DIM == 2
    h_particles->set(numParticles, numNodes, h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay);
    ParticlesNS::Kernel::Launch::set(d_particles, numParticles, numNodes, h_mass, h_x, h_y, h_vx, h_vy, h_ax, h_ay);
#else
    h_particles->set(&numParticles, &numNodes, h_mass, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_ax, h_ay, h_az);
    ParticlesNS::Kernel::Launch::set(d_particles, &numParticles, &numNodes, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                     d_ax, d_ay, d_az);
#endif

}

ParticleHandler::~ParticleHandler() {

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
    delete h_particles;

    //TODO: why is this not working (or necessary)?
    // device particle entries
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
    gpuErrorcheck(cudaFree(d_particles));

}

void ParticleHandler::positionToDevice() {
    gpuErrorcheck(cudaMemcpy(d_x,  h_x,  numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(d_y,  h_y,  numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(d_z,  h_z,  numParticles*sizeof(real), cudaMemcpyHostToDevice));
#endif
#endif
}
void ParticleHandler::velocityToDevice() {
    gpuErrorcheck(cudaMemcpy(d_vx, h_vx, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(d_vy, h_vy, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(d_vz, h_vz, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#endif
#endif
}
void ParticleHandler::accelerationToDevice() {
    gpuErrorcheck(cudaMemcpy(d_ax, h_ax, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(d_ay, h_ay, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(d_az, h_az, numParticles*sizeof(real), cudaMemcpyHostToDevice));
#endif
#endif
}

void ParticleHandler::distributionToDevice(bool velocity, bool acceleration) {

    positionToDevice();
    if (velocity) {
        velocityToDevice();
    }
    if (acceleration) {
        accelerationToDevice();
    }

}

void ParticleHandler::positionToHost() {
    gpuErrorcheck(cudaMemcpy(h_x, d_x, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(h_y, d_y, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(h_z, d_z, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#endif
#endif
}
void ParticleHandler::velocityToHost() {
    gpuErrorcheck(cudaMemcpy(h_vx, d_vx, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(h_vy, d_vy, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(h_vz, d_vz, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#endif
#endif
}
void ParticleHandler::accelerationToHost() {
    gpuErrorcheck(cudaMemcpy(h_x, d_x, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM > 1
    gpuErrorcheck(cudaMemcpy(h_y, d_y, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#if DIM == 3
    gpuErrorcheck(cudaMemcpy(h_z, d_z, numParticles*sizeof(real), cudaMemcpyDeviceToHost));
#endif
#endif
}

void ParticleHandler::distributionToHost(bool velocity, bool acceleration) {

    positionToHost();
    if (velocity) {
        velocityToHost();
    }
    if (acceleration) {
        accelerationToDevice();
    }

}
