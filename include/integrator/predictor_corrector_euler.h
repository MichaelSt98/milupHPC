/**
 * @file predictor_corrector_euler.h
 * @brief Predictor Corrector Euler integrator.
 *
 * Predictor Corrector Euler integrator inheriting from the Miluphpc class.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_PREDICTOR_CORRECTOR_EULER_H
#define MILUPHPC_PREDICTOR_CORRECTOR_EULER_H

#include "../miluphpc.h"
#include "device_predictor_corrector_euler.cuh"
#include "device_explicit_euler.cuh"


class PredictorCorrectorEuler : public Miluphpc {

private:

public:

    int device;
    struct cudaDeviceProp prop;
    int *d_blockCount;

    real *d_block_forces;
    real *d_block_courant;
    real *d_block_artVisc;
    real *d_block_e;
    real *d_block_rho;
    real *d_block_vmax;

    PredictorCorrectorEulerNS::BlockShared *d_blockShared;

    /**
     * @brief Constructor.
     *
     * @param simulationParameters Simulation parameters/settings.
     */
    PredictorCorrectorEuler(SimulationParameters simulationParameters);

    /**
     * @brief Destructor.
     */
    ~PredictorCorrectorEuler();

    /**
     * @brief Implementation of the abstract integration method.
     *
     * @param step Integration step (number)
     */
    void integrate(int step);
};

#endif //MILUPHPC_PREDICTOR_CORRECTOR_EULER_H
