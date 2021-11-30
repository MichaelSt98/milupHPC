#ifndef MILUPHPC_CUDAUTILITIES_CUH
#define MILUPHPC_CUDAUTILITIES_CUH

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../parameter.h"
#include <assert.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


#define safeCudaCall(call) checkCudaCall(call, #call, __FILE__, __LINE__)
#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * Somewhat of an assertion,
 * printing warning and/or terminating in dependence of `SAFETY_LEVEL`
 */
#if SAFETY_LEVEL == 0
#define cudaAssert(...)
#elif SAFETY_LEVEL == 1
#define cudaAssert(...) {    \
    printf(__VA_ARGS__);     \
}
#elif SAFETY_LEVEL == 2
#define cudaAssert(...) {    \
    printf(__VA_ARGS__);             \
    assert(0);               \
}
#elif SAFETY_LEVEL == 3
#define cudaAssert(...) {    \
    printf(__VA_ARGS__);             \
    assert(0);               \
}
#else
#define cudaAssert(...)
#endif

/**
 * Terminate from within CUDA kernel using assert(0)
 */
#define cudaTerminate(...) { \
    printf(__VA_ARGS__);     \
    assert(0);               \
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val);
#endif

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void checkCudaCall(cudaError_t command, const char * commandName, const char * fileName, int line);

namespace CudaUtils {
    namespace Kernel {

        __global__ void collectValues(integer *indices, real *entries, real *collector, integer count);

        __global__ void checkValues(integer *indices, real *entry1, real *entry2, real *entry3, integer count);

        template<typename T>
        __global__ void findDuplicates(T *array, integer *duplicateCounter, int length);

        template<typename T>
        __global__ void findDuplicateEntries(T *array1, T *array2, integer *duplicateCounter, int length);

        template<typename T>
        __global__ void findDuplicateEntries(T *array1, T *array2, T *array3, integer *duplicateCounter, int length);

        template<typename T, typename U>
        __global__ void findDuplicates(T *array, U *entry1, U *entry2, integer *duplicateCounter, int length);

        template<typename T>
        __global__ void markDuplicates(T *array, integer *duplicateCounter, int length);

        //template<typename T, typename U>
        //__global__ void markDuplicates(T *array, U *entry1, U *entry2, integer *duplicateCounter, int length);

        template<typename T, typename U>
        __global__ void markDuplicates(T *array, U *entry1, U *entry2, U *entry3, integer *duplicateCounter, integer *child, int length);

        template<typename T>
        __global__ void removeDuplicates(T *array, T *removedArray, integer *duplicateCounter, int length);

        namespace Launch {
            real collectValues(integer *indices, real *entries, real *collector, integer count);

            real checkValues(integer *indices, real *entry1, real *entry2, real *entry3, integer count);

            template<typename T>
            real findDuplicates(T *array, integer *duplicateCounter, int length);

            template<typename T>
            real findDuplicateEntries(T *array1, T *array2, integer *duplicateCounter, int length);

            template<typename T>
            real findDuplicateEntries(T *array1, T *array2, T *array3, integer *duplicateCounter, int length);

            template<typename T, typename U>
            real findDuplicates(T *array, U *entry1, U *entry2, integer *duplicateCounter, int length);

            template<typename T>
            real markDuplicates(T *array, integer *duplicateCounter, int length);

            //template<typename T, typename U>
            //real markDuplicates(T *array, U *entry1, U *entry2, integer *duplicateCounter, int length);

            template<typename T, typename U>
            real markDuplicates(T *array, U *entry1, U *entry2, U *entry3, integer *duplicateCounter, integer *child, int length);

            template<typename T>
            real removeDuplicates(T *array, T *removedArray, integer *duplicateCounter, int length);

        }
    }

}

namespace cuda {
    namespace math {
        __device__ real min(real a, real b);

        __device__ real min(real a, real b, real c);

        __device__ real max(real a, real b);

        __device__ real max(real a, real b, real c);

        __device__ real abs(real a);

        __device__ real sqrt(real a);

        __device__ real rsqrt(real a);
    }
}

#endif //MILUPHPC_CUDAUTILITIES_CUH
