#include "../include/miluphpc.h"

Miluphpc::Miluphpc(integer numParticles, integer numNodes) : numParticles(numParticles), numNodes(numNodes) {

    particleHandler = new ParticleHandler(numParticles, numNodes);
    subDomainKeyTreeHandler = new SubDomainKeyTreeHandler();

}

Miluphpc::~Miluphpc() {

    delete particleHandler;
    delete subDomainKeyTreeHandler;

}

void Miluphpc::initDistribution(ParticleDistribution::Type particleDistribution) {

    switch(particleDistribution) {
        case ParticleDistribution::disk:
            diskModel();
            break;
        case ParticleDistribution::plummer:
            //
            break;
        default:
            diskModel();
    }
}

void Miluphpc::diskModel() {

    real a = 1.0;
    real pi = 3.14159265;
    std::default_random_engine generator;
    std::uniform_real_distribution<real> distribution(1.5, 12.0);
    std::uniform_real_distribution<real> distribution_theta(0.0, 2 * pi);

    real solarMass = 100000;

    // loop through all particles
    for (int i = 0; i < numParticlesLocal; i++) {

        real theta = distribution_theta(generator);
        real r = distribution(generator);

        // set mass and position of particle
        if (subDomainKeyTreeHandler->h_subDomainKeyTree->rank == 0) {
            if (i == 0) {
                particleHandler->h_particles->mass[i] = 2 * solarMass / numParticles; //solarMass; //100000; 2 * solarMass / numParticles;
                particleHandler->h_particles->x[i] = 0;
                particleHandler->h_particles->y[i] = 0;
                particleHandler->h_particles->z[i] = 0;
            } else {
                particleHandler->h_particles->mass[i] = 2 * solarMass / numParticles;
                particleHandler->h_particles->x[i] = r * cos(theta);
                //y[i] = r * sin(theta);
                particleHandler->h_particles->z[i] = r * sin(theta);

                if (i % 2 == 0) {
                    particleHandler->h_particles->y[i] = i * 1e-7;//z[i] = i * 1e-7;
                } else {
                    particleHandler->h_particles->y[i] = i * -1e-7;//z[i] = i * -1e-7;
                }
            }
        }
        else {
            particleHandler->h_particles->mass[i] = 2 * solarMass / numParticles;
            particleHandler->h_particles->x[i] = (r + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 1.1e-1) *
                    cos(theta) + 1.0e-2*subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
            //y[i] = (r + h_subDomainHandler->rank * 1.3e-1) * sin(theta) + 1.1e-2*h_subDomainHandler->rank;
            particleHandler->h_particles->z[i] = (r + subDomainKeyTreeHandler->h_subDomainKeyTree->rank * 1.3e-1) *
                    sin(theta) + 1.1e-2*subDomainKeyTreeHandler->h_subDomainKeyTree->rank;

            if (i % 2 == 0) {
                //z[i] = i * 1e-7 * h_subDomainHandler->rank + 0.5e-7*h_subDomainHandler->rank;
                particleHandler->h_particles->y[i] = i * 1e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank +
                        0.5e-7*subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
            } else {
                //z[i] = i * -1e-7 * h_subDomainHandler->rank + 0.4e-7*h_subDomainHandler->rank;
                particleHandler->h_particles->y[i] = i * -1e-7 * subDomainKeyTreeHandler->h_subDomainKeyTree->rank
                        + 0.4e-7*subDomainKeyTreeHandler->h_subDomainKeyTree->rank;
            }
        }


        // set velocity of particle
        real rotation = 1;  // 1: clockwise   -1: counter-clockwise
        real v = sqrt(solarMass / (r));

        if (i == 0) {
            particleHandler->h_particles->vx[0] = 0.0;
            particleHandler->h_particles->vy[0] = 0.0;
            particleHandler->h_particles->vz[0] = 0.0;
        }
        else{
            particleHandler->h_particles->vx[i] = rotation*v*sin(theta);
            //y_vel[i] = -rotation*v*cos(theta);
            particleHandler->h_particles->vz[i] = -rotation*v*cos(theta);
            //z_vel[i] = 0.0;
            particleHandler->h_particles->vy[i] = 0.0;
        }

        // set acceleration to zero
        particleHandler->h_particles->ax[i] = 0.0;
        particleHandler->h_particles->ay[i] = 0.0;
        particleHandler->h_particles->az[i] = 0.0;
    }

}