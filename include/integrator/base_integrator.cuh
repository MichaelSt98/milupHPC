#ifndef MILUPHPC_BASE_INTEGRATOR_CUH
#define MILUPHPC_BASE_INTEGRATOR_CUH

#include "../particles.cuh"

class BaseIntegrator {

public:

    IntegratedParticles *integratedParticles;

    BaseIntegrator();
    ~BaseIntegrator();

    virtual void integrate() {};

    void rhs();

};

#endif //MILUPHPC_BASE_INTEGRATOR_CUH
