/**
 * @file leapfrog.h
 * @brief Leapfrog integrator.
 *
 * Verlet Strömer inheriting from the Miluphpc class.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_LEAPFROG_H
#define MILUPHPC_LEAPFROG_H

#include "../miluphpc.h"
#include "device_leapfrog.cuh"

class Leapfrog : public Miluphpc {

public:

    /**
     * @brief Constructor.
     *
     * @param simulationParameters Simulation parameters/settings.
     */
    Leapfrog(SimulationParameters simulationParameters);

    /**
     * @brief Destructor.
     */
    ~Leapfrog();

    /**
     * @brief Implementation of the abstract integration method.
     *
     * @param step Integration step (number)
     */
    void integrate(int step);

};


#endif //MILUPHPC_LEAPFROG_H
