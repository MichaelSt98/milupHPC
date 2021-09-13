#include "../../include/integrator/euler.h"

Euler::Euler(integer numParticles, integer numNodes) : Miluphpc(numParticles, numNodes) {
    integratedParticles = new IntegratedParticles[1];
    printf("Euler()\n");
}

Euler::~Euler() {
    printf("~Euler()\n");
}

void Euler::integrate() {
    printf("Euler::integrate()\n");
    rhs();
    //rhs();
}