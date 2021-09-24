#include "../../include/integrator/explicit_euler.h"

ExplicitEuler::ExplicitEuler(integer numParticles, integer numNodes) : Miluphpc(numParticles, numNodes) {
    //integratedParticles = new IntegratedParticles[1];
    printf("ExplicitEuler()\n");
}

ExplicitEuler::~ExplicitEuler() {
    printf("~ExplicitEuler()\n");
}

void ExplicitEuler::integrate() {
    printf("Euler::integrate()\n");
    rhs();
    Gravity::Kernel::Launch::update(particleHandler->d_particles, numParticlesLocal, 0.005, 1.);
}