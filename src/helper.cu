//
// Created by Michael Staneker on 15.08.21.
//

#include "../include/helper.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"
#include <cub/cub.cuh>

CUDA_CALLABLE_MEMBER Helper::Helper() {

}

CUDA_CALLABLE_MEMBER Helper::Helper(integer *integerBuffer, real *realBuffer) : integerBuffer(integerBuffer),
                realBuffer(realBuffer) {

}

CUDA_CALLABLE_MEMBER Helper::~Helper() {

}

CUDA_CALLABLE_MEMBER void Helper::set(integer *integerBuffer, real *realBuffer) {
    this->integerBuffer = integerBuffer;
    this->realBuffer = realBuffer;
}

namespace HelperNS {

    namespace Kernel {
        __global__ void set(Helper *helper, integer *integerBuffer, real *realBuffer) {
            helper->set(integerBuffer, realBuffer);
        }

        void Launch::set(Helper *helper, integer *integerBuffer, real *realBuffer) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::HelperNS::Kernel::set, helper, integerBuffer,
                         realBuffer);

        }
    }
}

namespace HelperNS {

    template <typename A, typename B>
    real sortArray(A *arrayToSort, A *sortedArray, B *keyIn, B *keyOut, integer n) {

        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        gpuErrorcheck(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                      keyIn, keyOut, arrayToSort, sortedArray, n));
        // Allocate temporary storage
        gpuErrorcheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        gpuErrorcheck(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes));

        // Run sorting operation
        gpuErrorcheck(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                      keyIn, keyOut, arrayToSort, sortedArray, n));

        gpuErrorcheck(cudaFree(d_temp_storage));

        return 0.f;
    }

    template real sortArray<real, integer>(real *arrayToSort, real *sortedArray, integer *keyIn, integer *keyOut,
            integer n);
    template real sortArray<real, keyType>(real *arrayToSort, real *sortedArray, keyType *keyIn, keyType *keyOut,
            integer n);
    template real sortArray<integer, integer>(integer *arrayToSort, integer *sortedArray, integer *keyIn,
            integer *keyOut, integer n);
    template real sortArray<integer, keyType>(integer *arrayToSort, integer *sortedArray, keyType *keyIn,
            keyType *keyOut, integer n);
    template real sortArray<keyType, integer>(keyType *arrayToSort, keyType *sortedArray, integer *keyIn,
            integer *keyOut, integer n);
    template real sortArray<keyType , keyType>(keyType *arrayToSort, keyType *sortedArray, keyType *keyIn,
            keyType *keyOut, integer n);

}