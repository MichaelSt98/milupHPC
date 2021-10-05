#ifndef MILUPHPC_EXPLICIT_EULER_H
#define MILUPHPC_EXPLICIT_EULER_H

#include "../miluphpc.h"

class ExplicitEuler : public Miluphpc {

public:

    ExplicitEuler(SimulationParameters simulationParameters, integer numParticles, integer numNodes);
    ~ExplicitEuler();

    void integrate(int step);

};


#endif //MILUPHPC_EXPLICIT_EULER_H
