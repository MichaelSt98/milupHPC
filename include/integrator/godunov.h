/**
 * @file godunov.h
 * @brief Godunov's method integrator.
 *
 * Godunov's method integrator used for meshless schemes MFV/MFM inheriting from the Miluphpc class.
 *
 * @author Johannes S. Martin
 * @bug no known bugs
 */

#ifndef MILUPHPC_GODUNOV_H
#define MILUPHPC_GODUNOV_H

#include "../miluphpc.h"
#include "device_godunov.cuh"

class Godunov : public Miluphpc {

public:

    int device;
    struct cudaDeviceProp prop;
    int *d_blockCount;

    real *d_block_dt;

    /**
     * @brief Constructor.
     *
     * @param simulationParameters Simulation parameters/settings.
     */
    Godunov(SimulationParameters simulationParameters);

    /**
     * @brief Destructor.
     */
     ~Godunov();

     /**
      * @brief Implementation of the abstract integration method.
      *
      * @param step Integration step (number).
      */
      void integrate(int step);

};

#endif //MILUPHPC_GODUNOV_H
