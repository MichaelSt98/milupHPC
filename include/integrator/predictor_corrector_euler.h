#ifndef MILUPHPC_PREDICTOR_CORRECTOR_EULER_H
#define MILUPHPC_PREDICTOR_CORRECTOR_EULER_H

#include "../miluphpc.h"
#include "device_predictor_corrector_euler.cuh"


class PredictorCorrectorEuler : public Miluphpc {

private:

    int *device;
    struct cudaDeviceProp *prop;
    int *d_blockCount;

    real *d_block_forces;
    real *d_block_courant;
    real *d_block_artVisc;
    real *d_block_e;
    real *d_block_rho;

public:

    PredictorCorrectorEulerNS::BlockShared *d_blockShared;

    PredictorCorrectorEuler(SimulationParameters simulationParameters);
    ~PredictorCorrectorEuler();

    void integrate(int step);
};


#endif //MILUPHPC_PREDICTOR_CORRECTOR_EULER_H
