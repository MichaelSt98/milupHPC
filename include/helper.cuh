#ifndef MILUPHPC_HELPER_CUH
#define MILUPHPC_HELPER_CUH

#include "parameter.h"
#include "cuda_utils/cuda_utilities.cuh"
#include <boost/mpi.hpp>
#include "utils/logger.h"
#include "cuda_utils/cuda_runtime.h"

struct Reduction
{
    enum Type
    {
        min, max, sum
    };
    Type t_;
    Reduction(Type t) : t_(t) {}
    operator Reduction () const {return t_;}
private:
    template<typename T>
    operator T () const;
};

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

        template <typename T>
        __global__ void resetArray(T *array, T value, integer n);

        namespace Launch {
            template <typename T>
            real copyArray(T *targetArray, T *sourceArray, integer n);

            template <typename T>
            real resetArray(T *array, T value, integer n);
        }
    }

    template <typename A>
    real sortKeys(A *keysToSort, A *sortedKeys, int n);

    template <typename A, typename B>
    real sortArray(A *arrayToSort, A *sortedArray, B *keyIn, B *keyOut, integer n);

    template <typename T>
    T reduceAndGlobalize(T *d_sml, T *d_aggregate, integer n, Reduction::Type reductionType);

}

#endif //MILUPHPC_HELPER_CUH
