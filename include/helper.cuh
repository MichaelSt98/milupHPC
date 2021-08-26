//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_HELPER_CUH
#define MILUPHPC_HELPER_CUH

#include "parameter.h"
#include "cuda_utils/cuda_utilities.cuh"

class Helper {

public:

    integer *integerBuffer;
    real *realBuffer;

    CUDA_CALLABLE_MEMBER Helper();
    CUDA_CALLABLE_MEMBER Helper(integer *integerBuffer, real *realBuffer);
    CUDA_CALLABLE_MEMBER ~Helper();
    CUDA_CALLABLE_MEMBER void set(integer *integerBuffer, real *realBuffer);

};

namespace HelperNS {

    namespace Kernel {
        __global__ void set(Helper *helper, integer *integerBuffer, real *realBuffer);

        namespace Launch {
            void set(Helper *helper, integer *integerBuffer, real *realBuffer);
        }
    }

    template <typename A, typename B>
    real sortArray(A *arrayToSort, A *sortedArray, B *keyIn, B *keyOut, integer n);

}

#endif //MILUPHPC_HELPER_CUH
