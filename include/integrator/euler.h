#ifndef MILUPHPC_EULER_H
#define MILUPHPC_EULER_H

#include "../miluphpc.h"

class Euler : public Miluphpc {

public:

    Euler(SimulationParameters simulationParameters, integer numParticles, integer numNodes);
    ~Euler();

    void integrate(int step);

};

#endif //MILUPHPC_EULER_H