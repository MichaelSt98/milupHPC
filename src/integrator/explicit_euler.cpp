#include "../../include/integrator/explicit_euler.h"

ExplicitEuler::ExplicitEuler(SimulationParameters simulationParameters, integer numParticles,
                             integer numNodes) : Miluphpc(simulationParameters, numParticles, numNodes) {
    //integratedParticles = new IntegratedParticles[1];
    printf("ExplicitEuler()\n");
}

ExplicitEuler::~ExplicitEuler() {
    printf("~ExplicitEuler()\n");
}

void ExplicitEuler::integrate() {
    printf("Euler::integrate()\n");
    Timer timer;
    real time = rhs();
    Logger(TIME) << "rhs: " << time << " ms";

    Logger(TIME) << "rhs elapsed: " << timer.elapsed() << " ms";
    Gravity::Kernel::Launch::update(particleHandler->d_particles, numParticlesLocal,
                                    (real)simulationParameters.timestep, (real)simulationParameters.dampening);
}