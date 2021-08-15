//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_HELPER_CUH
#define MILUPHPC_HELPER_CUH

#include "parameter.h"
#include "cuda_utils/cuda_utilities.cuh"

class Helper {

public:

    integer *intBuffer;
    real *floatBuffer;

    CUDA_CALLABLE_MEMBER Helper();
    CUDA_CALLABLE_MEMBER Helper(integer *intBuffer, real *floatBuffer);
    CUDA_CALLABLE_MEMBER ~Helper();
    CUDA_CALLABLE_MEMBER void set(integer *intBuffer, real *floatBuffer);

};

namespace HelperNS {

    __global__ void setKernel(Helper *helper, integer *intBuffer, real *floatBuffer);

    void launchSetKernel(Helper *helper, integer *intBuffer, real *floatBuffer);
}

#endif //MILUPHPC_HELPER_CUH
