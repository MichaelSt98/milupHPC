#ifndef MILUPHPC_EXPLICIT_EULER_H
#define MILUPHPC_EXPLICIT_EULER_H

#include "../miluphpc.h"

class ExplicitEuler : public Miluphpc {

public:

    ExplicitEuler(integer numParticles, integer numNodes);
    ~ExplicitEuler();

    void integrate();

};


#endif //MILUPHPC_EXPLICIT_EULER_H
