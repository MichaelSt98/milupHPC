#ifndef MILUPHPC_PREDICTOR_CORRECTOR_H
#define MILUPHPC_PREDICTOR_CORRECTOR_H

#include "../miluphpc.h"

class PredictorCorrector : public Miluphpc {

public:
    //PredictorCorrector(SimulationParameters simulationParameters, integer numParticles, integer numNodes);
    PredictorCorrector(SimulationParameters simulationParameters);
    ~PredictorCorrector();

    void integrate(int step);

};

#endif //MILUPHPC_PREDICTOR_CORRECTOR_H
