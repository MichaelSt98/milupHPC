#ifndef MILUPHPC_PREDICTOR_CORRECTOR_EULER_H
#define MILUPHPC_PREDICTOR_CORRECTOR_EULER_H

#include "../miluphpc.h"
#include "device_predictor_corrector_euler.cuh"

class PredictorCorrectorEuler : public Miluphpc {

public:

    PredictorCorrectorEuler(SimulationParameters simulationParameters);
    ~PredictorCorrectorEuler();

    void integrate(int step);
};


#endif //MILUPHPC_PREDICTOR_CORRECTOR_EULER_H
