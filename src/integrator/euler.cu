#include "../../include/integrator/euler.cuh"

Euler::Euler() : BaseIntegrator() {
    integratedParticles = new IntegratedParticles[1];
    printf("Euler()\n");
}

Euler::~Euler() {
    printf("~Euler()\n");
}

void Euler::integrate() {
    rhs();
    printf("Euler::integrator()\n");
    rhs();
}