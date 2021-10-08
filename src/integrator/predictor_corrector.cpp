#include "../../include/integrator/predictor_corrector.h"

/*PredictorCorrector::PredictorCorrector(SimulationParameters simulationParameters, integer numParticles,
                                       integer numNodes) : Miluphpc(simulationParameters, numParticles, numNodes) {

    integratedParticles = new IntegratedParticles[2];
    printf("PredictorCorrector()\n");

}*/

PredictorCorrector::PredictorCorrector(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {
    integratedParticles = new IntegratedParticles[2];
    printf("PredictorCorrector()\n");
}

PredictorCorrector::~PredictorCorrector() {
    printf("~PredictorCorrector()\n");
}

void PredictorCorrector::integrate(int step) {
    printf("PredictorCorrector::integrate()\n");
    rhs(step);
    //rhs();
}