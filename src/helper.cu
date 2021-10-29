#include "../include/helper.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"
#include <cub/cub.cuh>

CUDA_CALLABLE_MEMBER Helper::Helper() {

}

CUDA_CALLABLE_MEMBER Helper::Helper(integer *integerVal, real *realVal, keyType *keyTypeVal, integer *integerBuffer,
                                    real *realBuffer, keyType *keyTypeBuffer) : integerVal(integerVal),
                                    realVal(realVal), keyTypeVal(keyTypeVal), integerBuffer(integerBuffer),
                                    realBuffer(realBuffer) , keyTypeBuffer(keyTypeBuffer) {

}

CUDA_CALLABLE_MEMBER Helper::~Helper() {

}

CUDA_CALLABLE_MEMBER void Helper::set(integer *integerVal, real *realVal, keyType *keyTypeVal, integer *integerBuffer,
                                      real *realBuffer, keyType *keyTypeBuffer) {
    this->integerVal = integerVal;
    this->realVal = realVal;
    this->keyTypeVal = keyTypeVal;
    this->integerBuffer = integerBuffer;
    this->realBuffer = realBuffer;
    this->keyTypeBuffer = keyTypeBuffer;
}

namespace HelperNS {

    namespace Kernel {
        __global__ void set(Helper *helper, integer *integerVal, real *realVal, keyType *keyTypeVal,
                            integer *integerBuffer, real *realBuffer, keyType *keyTypeBuffer) {
            helper->set(integerVal, realVal, keyTypeVal, integerBuffer, realBuffer, keyTypeBuffer);
        }

        void Launch::set(Helper *helper, integer *integerVal, real *realVal, keyType *keyTypeVal,
                         integer *integerBuffer, real *realBuffer, keyType *keyTypeBuffer) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::HelperNS::Kernel::set, helper, integerVal, realVal, keyTypeVal,
                         integerBuffer, realBuffer, keyTypeBuffer);

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
        cuda::malloc(d_temp_storage, temp_storage_bytes);

        // Run sorting operation
        gpuErrorcheck(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                      keyIn, keyOut, arrayToSort, sortedArray, n));

        cuda::free(d_temp_storage);

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


    template <typename T>
    T reduceAndGlobalize(T *d_sml, T *d_aggregate, integer n, Reduction::Type reductionType) {

        // device wide reduction
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        switch (reductionType) {
            case Reduction::min: {
                cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_sml, d_aggregate, n);
                // Allocate temporary storage
                cuda::malloc(d_temp_storage, temp_storage_bytes);
                // Run max-reduction
                cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_sml, d_aggregate, n);
            } break;
            case Reduction::max: {
                cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_sml, d_aggregate, n);
                // Allocate temporary storage
                cuda::malloc(d_temp_storage, temp_storage_bytes);
                // Run max-reduction
                cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_sml, d_aggregate, n);
            } break;
            case Reduction::sum: {
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sml, d_aggregate, n);
                // Allocate temporary storage
                cuda::malloc(d_temp_storage, temp_storage_bytes);
                // Run max-reduction
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sml, d_aggregate, n);
            } break;
            default: {
                Logger(ERROR) << "Reduction type not available!";
            }
        }

        T reduction;
        gpuErrorcheck(cudaMemcpy(&reduction, d_aggregate, sizeof(T), cudaMemcpyDeviceToHost));
        Logger(INFO) << "reduction = " << reduction;

        switch (reductionType) {
            case Reduction::min: {
                // interprocess reduction
                boost::mpi::communicator comm;
                all_reduce(comm, boost::mpi::inplace_t<T *>(&reduction), 1, boost::mpi::minimum<T>());
            } break;
            case Reduction::max: {
                // interprocess reduction
                boost::mpi::communicator comm;
                all_reduce(comm, boost::mpi::inplace_t<T *>(&reduction), 1, boost::mpi::maximum<T>());
            } break;
            case Reduction::sum: {
                // interprocess reduction
                boost::mpi::communicator comm;
                all_reduce(comm, boost::mpi::inplace_t<T *>(&reduction), 1, std::plus<T>());
            } break;
            default: {
                Logger(ERROR) << "Reduction type not available!";
            }
        }
        Logger(INFO) << "globalized reduction = " << reduction;

        cuda::free(d_temp_storage);

        return reduction;

    }

    template real reduceAndGlobalize<real>(real*, real*, integer, Reduction::Type);


    namespace Kernel {

        template <typename T>
        __global__ void copyArray(T *targetArray, T *sourceArray, integer n) {

            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            while ((index + offset) < n) {
                targetArray[index + offset] = sourceArray[index + offset];

                offset += stride;
            }
        }

        template <typename T>
        __global__ void resetArray(T *array, T value, integer n) {

            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            while ((index + offset) < n) {
                array[index + offset] = value;

                offset += stride;
            }
        }

        namespace Launch {

            template<typename T>
            real copyArray(T *targetArray, T *sourceArray, integer n) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::HelperNS::Kernel::copyArray, targetArray, sourceArray, n);
            }
            template real copyArray<integer>(integer *targetArray, integer *sourceArray, integer n);
            template real copyArray<real>(real *targetArray, real *sourceArray, integer n);
            template real copyArray<keyType>(keyType *targetArray, keyType *sourceArray, integer n);

            template <typename T>
            real resetArray(T *array, T value, integer n) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::HelperNS::Kernel::resetArray, array, value, n);
            }
            template real resetArray<integer>(integer *array, integer value, integer n);
            //template real resetArray<idInteger>(idInteger *array, idInteger value, integer n);
            template real resetArray<real>(real *array, real value, integer n);
            template real resetArray<keyType>(keyType *array, keyType value, integer n);

        }
        /*__global__ void reset(Helper *helper, int length) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((index + offset) < length) {
                helper->
            }
        }*/
    }

}
