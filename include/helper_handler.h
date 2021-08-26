#ifndef MILUPHPC_HELPER_HANDLER_H
#define MILUPHPC_HELPER_HANDLER_H

#include "helper.cuh"
#include "parameter.h"

class HelperHandler {

    integer length;

    integer *d_integerBuffer;
    real *d_realBuffer;

    Helper *d_helper;

    HelperHandler(integer length);
    ~HelperHandler();

};


#endif //MILUPHPC_HELPER_HANDLER_H
