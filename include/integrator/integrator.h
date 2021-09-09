#ifndef MILUPHPC_INTEGRATOR_H
#define MILUPHPC_INTEGRATOR_H

#include "predictor_corrector.cuh"
#include "euler.cuh"
#include "../parameter.h"

#include <iostream>
#include <stdio.h>

class Integrator {

public:

    BaseIntegrator *integrator;

    Integrator(IntegratorSelection::Type integratorSelection=IntegratorSelection::euler);
    ~Integrator();

    void integrate();

};


#endif //MILUPHPC_INTEGRATOR_H
