#ifndef MILUPHPC_PREDICTOR_CORRECTOR_CUH
#define MILUPHPC_PREDICTOR_CORRECTOR_CUH

#include "base_integrator.cuh"

class PredictorCorrector : public BaseIntegrator {

public:
    PredictorCorrector();
    ~PredictorCorrector();

    void integrate();

};

#endif //MILUPHPC_PREDICTOR_CORRECTOR_CUH
