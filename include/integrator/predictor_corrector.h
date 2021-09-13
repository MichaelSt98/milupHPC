#ifndef MILUPHPC_PREDICTOR_CORRECTOR_H
#define MILUPHPC_PREDICTOR_CORRECTOR_H

#include "../miluphpc.h"

class PredictorCorrector : public Miluphpc {

public:
    PredictorCorrector(integer numParticles, integer numNodes);
    ~PredictorCorrector();

    void integrate();

};

#endif //MILUPHPC_PREDICTOR_CORRECTOR_H
