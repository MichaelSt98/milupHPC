#ifndef MILUPHPC_HELPER_HANDLER_H
#define MILUPHPC_HELPER_HANDLER_H

#include "helper.cuh"
#include "parameter.h"

#include <mpi.h>

class HelperHandler {

public:
    integer length;

    integer *d_integerVal;
    real *d_realVal;
    keyType  *d_keyTypeVal;

    integer *d_integerBuffer;
    real *d_realBuffer;
    keyType *d_keyTypeBuffer;

    Helper *d_helper;

    HelperHandler(integer length);
    ~HelperHandler();

    void reset();

};




#endif //MILUPHPC_HELPER_HANDLER_H
