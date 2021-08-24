#include "../../include/integrator/predictor_corrector.cuh"

PredictorCorrector::PredictorCorrector() : BaseIntegrator() {
    integratedParticles = new IntegratedParticles[2];
    printf("PredictorCorrector()\n");
}

PredictorCorrector::~PredictorCorrector() {
    printf("~PredictorCorrector()\n");
}

void PredictorCorrector::integrate() {
    rhs();
    printf("PredictorCorrector::integrator()\n");
    rhs();
}