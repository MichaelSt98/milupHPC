#ifndef MILUPHPC_EULER_CUH
#define MILUPHPC_EULER_CUH

#include "base_integrator.cuh"

class Euler : public BaseIntegrator {

public:
    Euler();
    ~Euler();

    void integrate();

};

#endif //MILUPHPC_EULER_CUH