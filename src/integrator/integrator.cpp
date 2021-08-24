//
// Created by Michael Staneker on 24.08.21.
//

#include "../../include/integrator/integrator.h"

Integrator::Integrator(IntegratorSelection::Type integratorSelection) {
    switch(integratorSelection) {
        case IntegratorSelection::euler:
            integrator = new Euler();
            break;
        case IntegratorSelection::predictor_corrector:
            integrator = new PredictorCorrector();
            break;
        default:
            printf("not available!");
    }
}

Integrator::~Integrator() {
    delete integrator;
}

void Integrator::integrate() {
    integrator->integrate();
}
