#include "../../include/integrator/euler.h"

Euler::Euler(SimulationParameters simulationParameters, integer numParticles,
             integer numNodes) : Miluphpc(simulationParameters, numParticles, numNodes) {

    integratedParticles = new IntegratedParticles[1];
    printf("Euler()\n");

}

Euler::~Euler() {
    printf("~Euler()\n");
}

void Euler::integrate(int step) {
    printf("Euler::integrate()\n");
    rhs(step);
    //rhs();
}