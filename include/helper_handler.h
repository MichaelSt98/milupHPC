#ifndef MILUPHPC_HELPER_HANDLER_H
#define MILUPHPC_HELPER_HANDLER_H

#include "helper.cuh"
#include "parameter.h"

#include <mpi.h>

class HelperHandler {

public:
    //integer length;

    int numProcesses;
    int numParticlesLocal;
    int numParticles;
    int sumParticles;
    int numNodes;

    integer *d_integerVal;
    integer *d_integerVal1;
    integer *d_integerVal2;

    real *d_realVal;
    real *d_realVal1;
    real *d_realVal2;

    keyType  *d_keyTypeVal;

    integer *d_integerBuffer;
    integer *d_integerBuffer1; // numParticles (or numParticlesLocal)
    integer *d_integerBuffer2; // numParticles (or numParticlesLocal)
    integer *d_integerBuffer3; // numParticles (or numParticlesLocal)
    integer *d_integerBuffer4; // numParticles (or numParticlesLocal)

    integer *d_sendCount; // subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses
    integer *d_sendCount1; // subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses

    idInteger *d_idIntegerBuffer;
    idInteger *d_idIntegerBuffer1;

    real *d_realBuffer;
    real *d_realBuffer1;

    keyType *d_keyTypeBuffer; // numParticlesLocal
    keyType *d_keyTypeBuffer1; // sumParticles
    keyType *d_keyTypeBuffer2; //sumParticles

    Helper *d_helper;

    HelperHandler(int numProcesses, int numParticlesLocal, int numParticles, int sumParticles, int numNodes);
    ~HelperHandler();

    void reset();

};




#endif //MILUPHPC_HELPER_HANDLER_H
