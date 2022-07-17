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

    /*integer *integerBuffer;
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

  */

    integer *integerVal;
    integer *integerVal1;
    integer *integerVal2;

    real *realVal;
    real *realVal1;
    real *realVal2;

    keyType  *keyTypeVal;

    integer *integerBuffer;
    integer *integerBuffer1; // numParticles (or numParticlesLocal)
    integer *integerBuffer2; // numParticles (or numParticlesLocal)
    integer *integerBuffer3; // numParticles (or numParticlesLocal)
    integer *integerBuffer4; // numParticles (or numParticlesLocal)

    integer *sendCount; // subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses
    integer *sendCount1; // subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses

    idInteger *idIntegerBuffer;
    idInteger *idIntegerBuffer1;

    real *realBuffer;
    real *realBuffer1;

    keyType *keyTypeBuffer; // numParticlesLocal
    keyType *keyTypeBuffer1; // sumParticles
    keyType *keyTypeBuffer2; //sumParticles

    CUDA_CALLABLE_MEMBER Helper();
    CUDA_CALLABLE_MEMBER Helper(integer *integerVal, integer *integerVal1, integer *integerVal2,
                                real *realVal, real *realVal1, real *realVal2, keyType *keyTypeVal,
                                integer *integerBuffer, integer *integerBuffer1, integer *integerBuffer2,
                                integer *integerBuffer3, integer *integerBuffer4,
                                integer *sendCount, integer *sendCount1, idInteger *idIntegerBuffer,
                                idInteger *idIntegerBuffer1, real *realBuffer, real *realBuffer1,
                                keyType *keyTypeBuffer, keyType *keyTypeBuffer1, keyType *keyTypeBuffer2);

    CUDA_CALLABLE_MEMBER void set(integer *integerVal, integer *integerVal1, integer *integerVal2,
                                  real *realVal, real *realVal1, real *realVal2, keyType *keyTypeVal,
                                  integer *integerBuffer, integer *integerBuffer1, integer *integerBuffer2,
                                  integer *integerBuffer3, integer *integerBuffer4,
                                  integer *sendCount, integer *sendCount1, idInteger *idIntegerBuffer,
                                  idInteger *idIntegerBuffer1, real *realBuffer, real *realBuffer1,
                                  keyType *keyTypeBuffer, keyType *keyTypeBuffer1, keyType *keyTypeBuffer2);

    CUDA_CALLABLE_MEMBER ~Helper();

};

#if TARGET_GPU
namespace HelperNS {

    namespace Kernel {
        //__global__ void set(Helper *helper, integer *integerVal, real *realVal, keyType *keyTypeVal,
        //                    integer *integerBuffer, real *realBuffer, keyType *keyTypeBuffer);

        __global__ void set(Helper *helper, integer *integerVal, integer *integerVal1, integer *integerVal2,
                            real *realVal, real *realVal1, real *realVal2, keyType *keyTypeVal,
                            integer *integerBuffer, integer *integerBuffer1, integer *integerBuffer2,
                            integer *integerBuffer3, integer *integerBuffer4,
                            integer *sendCount, integer *sendCount1, idInteger *idIntegerBuffer,
                            idInteger *idIntegerBuffer1, real *realBuffer, real *realBuffer1,
                            keyType *keyTypeBuffer, keyType *keyTypeBuffer1, keyType *keyTypeBuffer2);

        namespace Launch {
            //void set(Helper *helper, integer *integerVal, real *realVal, keyType *keyTypeVal, integer *integerBuffer,
            //         real *realBuffer, keyType *keyTypeBuffer);

            void set(Helper *helper, integer *integerVal, integer *integerVal1, integer *integerVal2,
                     real *realVal, real *realVal1, real *realVal2, keyType *keyTypeVal,
                     integer *integerBuffer, integer *integerBuffer1, integer *integerBuffer2,
                     integer *integerBuffer3, integer *integerBuffer4,
                     integer *sendCount, integer *sendCount1, idInteger *idIntegerBuffer,
                     idInteger *idIntegerBuffer1, real *realBuffer, real *realBuffer1,
                     keyType *keyTypeBuffer, keyType *keyTypeBuffer1, keyType *keyTypeBuffer2);
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
#endif // TARGET_GPU
#endif //MILUPHPC_HELPER_CUH
