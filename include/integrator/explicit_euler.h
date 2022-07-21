/**
 * @file explicit_euler.h
 * @brief Explicit Euler integrator.
 *
 * Explicit Euler integrator inheriting from the Miluphpc class.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_EXPLICIT_EULER_H
#define MILUPHPC_EXPLICIT_EULER_H

#include "../miluphpc.h"
#include "device_explicit_euler.cuh"

class ExplicitEuler : public Miluphpc {

public:

    //ExplicitEuler(SimulationParameters simulationParameters, integer numParticles, integer numNodes);
    /**
     * @brief Constructor.
     *
     * @param simulationParameters Simulation parameters/settings.
     */
    ExplicitEuler(SimulationParameters simulationParameters);

    /**
     * @brief Destructor.
     */
    ~ExplicitEuler();

    /**
     * @brief Implementation of the abstract integration method.
     *
     * @param step Integration step (number)
     */
    void integrate(int step);

    void update(Particles *particles, int numParticlesLocal, real dt);

};


#endif //MILUPHPC_EXPLICIT_EULER_H
