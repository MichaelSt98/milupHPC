/**
 * @file cuda_utilities.cuh
 * @brief CUDA utilities.
 *
 * CUDA utilities like:
 *
 * * preprocessor directives
 * * preprocessor macros
 * * CUDA math functions
 * * general CUDA kernels like
 *     * finding duplicates
 *     * copying values from device to device
 *     * ...
 *
 * @author Michael Staneker
 * @bug no
*/
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

/**
 * @brief check CUDA call
 */
#define safeCudaCall(call) checkCudaCall(call, #call, __FILE__, __LINE__)

/**
 * @brief check CUDA call
 */
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

/**
 * @brief Check CUDA error codes.
 *
 * @param code CUDA error code
 * @param file File
 * @param line Line
 * @param abort Abort in case of error
 */
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

/**
 * @brief Check CUDA call.
 *
 * @param command
 * @param commandName
 * @param fileName
 * @param line
 */
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
    /**
     * @brief Device/CUDA math functions.
     *
     * Wrapping CUDA math functionalities to provide correct function
     * for both single and double precision.
     */
    namespace math {
        /**
         * @brief Minimum value out of two floating point values.
         *
         * @param a Floating point value 1
         * @param b Floating point value 2
         * @return Minimum floating point value
         */
        __device__ real min(real a, real b);

        /**
         * @brief Minimum value out of three floating point values.
         *
         * @param a Floating point value 1
         * @param b Floating point value 2
         * @param c Floating point value 3
         * @return Minimum floating point value
         */
        __device__ real min(real a, real b, real c);

        /**
         * @brief Maximum value out of two floating point values.
         *
         * @param a Floating point value 1
         * @param b Floating point value 2
         * @return Maximum floating point value
         */
        __device__ real max(real a, real b);

        /**
         * @brief Maximum value out of three floating point values.
         *
         * @param a Floating point value 1
         * @param b Floating point value 2
         * @param c Floating point value 3
         * @return Maximum floating point value
         */
        __device__ real max(real a, real b, real c);

        /**
         * @brief Absolute value of a floating point value.
         *
         * @param a Floating point value
         * @return Absolute value of floating point value.
         */
        __device__ real abs(real a);

        /**
         * @brief Square root of a floating point value.
         *
         * @param a Floating point value
         * @return Square root of floating point value
         */
        __device__ real sqrt(real a);

        /**
         * @brief Inverse square root of a floating point value.
         *
         * @param a Floating point value
         * @return Inverse square root of a floating point value
         */
        __device__ real rsqrt(real a);
    }
}

#endif //MILUPHPC_CUDAUTILITIES_CUH
