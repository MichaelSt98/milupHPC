#ifndef MILUPHPC_CUDAUTILITIES_CUH
#define MILUPHPC_CUDAUTILITIES_CUH

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../parameter.h"


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


#define safeCudaCall(call) checkCudaCall(call, #call, __FILE__, __LINE__)
#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

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

#endif //MILUPHPC_CUDAUTILITIES_CUH
