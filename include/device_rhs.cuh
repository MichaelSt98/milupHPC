//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_DEVICE_RHS_CUH
#define MILUPHPC_DEVICE_RHS_CUH

#include "subdomain_key_tree/tree.cuh"
#include "particles.cuh"
#include "constants.h"

namespace Kernel {

    __global__ void resetArrays(Tree *tree, Particles *particles, integer *mutex, integer n, integer m);
    namespace Launch {
        real resetArrays(Tree *tree, Particles *particles, integer *mutex, integer n, integer m, bool time=false);
    }

}

#endif //MILUPHPC_DEVICE_RHS_CUH
