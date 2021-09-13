#ifndef MILUPHPC_EULER_H
#define MILUPHPC_EULER_H

#include "../miluphpc.h"

class Euler : public Miluphpc {

public:

    Euler(integer numParticles, integer numNodes);
    ~Euler();

    void integrate();

};

#endif //MILUPHPC_EULER_H