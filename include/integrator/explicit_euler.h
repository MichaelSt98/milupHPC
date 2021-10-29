#ifndef MILUPHPC_EXPLICIT_EULER_H
#define MILUPHPC_EXPLICIT_EULER_H

#include "../miluphpc.h"
#include "device_explicit_euler.cuh"

class ExplicitEuler : public Miluphpc {

public:

    //ExplicitEuler(SimulationParameters simulationParameters, integer numParticles, integer numNodes);
    ExplicitEuler(SimulationParameters simulationParameters);
    ~ExplicitEuler();

    void integrate(int step);

};


#endif //MILUPHPC_EXPLICIT_EULER_H
