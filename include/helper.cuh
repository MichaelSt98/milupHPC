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
    keyType *keyTypeBuffer;

    integer *integerVal;
    real *realVal;
    keyType *keyTypeVal;

    CUDA_CALLABLE_MEMBER Helper();
    CUDA_CALLABLE_MEMBER Helper(integer *integerVal, real *realVal, keyType *keyTypeVal, integer *integerBuffer,
                                real *realBuffer, keyType *keyTypeBuffer);
    CUDA_CALLABLE_MEMBER ~Helper();
    CUDA_CALLABLE_MEMBER void set(integer *integerVal, real *realVal, keyType *keyTypeVal, integer *integerBuffer,
                                  real *realBuffer, keyType *keyTypeBuffer);

};

namespace HelperNS {

    namespace Kernel {
        __global__ void set(Helper *helper, integer *integerVal, real *realVal, keyType *keyTypeVal,
                            integer *integerBuffer, real *realBuffer, keyType *keyTypeBuffer);

        namespace Launch {
            void set(Helper *helper, integer *integerVal, real *realVal, keyType *keyTypeVal, integer *integerBuffer,
                     real *realBuffer, keyType *keyTypeBuffer);
        }

        template <typename T>
        __global__ void copyArray(T *targetArray, T *sourceArray, integer n);

        namespace Launch {
            template <typename T>
            real copyArray(T *targetArray, T *sourceArray, integer n);
        }
    }

    template <typename A, typename B>
    real sortArray(A *arrayToSort, A *sortedArray, B *keyIn, B *keyOut, integer n);

}

#endif //MILUPHPC_HELPER_CUH
