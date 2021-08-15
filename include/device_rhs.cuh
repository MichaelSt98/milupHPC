//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_DEVICE_RHS_CUH
#define MILUPHPC_DEVICE_RHS_CUH

#include "subdomain_key_tree/tree.cuh"
#include "particles.cuh"
#include "constants.h"

namespace device {

    __global__ void resetArraysKernel(Tree *tree, Particles *particles, integer *mutex, integer n, integer m);
    void launchResetArraysKernel(Tree *tree, Particles *particles, integer *mutex, integer n, integer m);

}

#endif //MILUPHPC_DEVICE_RHS_CUH
