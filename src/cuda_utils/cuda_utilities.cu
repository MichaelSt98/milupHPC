#include "../../include/cuda_utils/cuda_utilities.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
    }
}

void checkCudaCall(cudaError_t command, const char * commandName, const char * fileName, int line)
{
    if (command != cudaSuccess)
    {
        fprintf(stderr, "Error: CUDA result \"%s\" for call \"%s\" in file \"%s\" at line %d. Terminating...\n",
                cudaGetErrorString(command), commandName, fileName, line);
        exit(0);
    }
}


namespace CudaUtils {

    namespace Kernel {

        __global__ void collectValues(integer *indices, real *entries, real *collector, integer count) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((index + offset) < count) {
                collector[index + offset] = entries[indices[index + offset]];
                offset += stride;
            }
        }

        template<typename T>
        __global__ void findDuplicates(T *array, integer *duplicateCounter, int length) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((index + offset) < length) {
                for (int i=0; i<length; i++) {
                    /*if (i != (index + offset)) {
                        if (array[index + offset] == array[i]) {
                            duplicateCounter += 1;
                        }
                    }*/
                }
                offset += stride;
            }
        }

        template<typename T>
        __global__ void markDuplicates(T *array, integer *duplicateCounter, int length) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            integer maxIndex;

            //remark: check only x, but in principle check all
            while ((index + offset) < length) {
                if (array[index + offset] != -1) {
                    for (integer i = 0; i < length; i++) {
                        if (i != (index + offset)) {
                            if (array[i] != -1 &&  array[index + offset] == array[i]) {
                                maxIndex = max(index + offset, i);
                                // mark larger index with -1 (thus a duplicate)
                                array[maxIndex] = -1;
                                atomicAdd(duplicateCounter, 1);
                            }
                        }

                    }
                }
                offset += stride;
            }
        }

        template<typename T>
        __global__ void removeDuplicates(T *array, T *removedArray, integer *duplicateCounter, int length) {

            int index = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int offset = 0;

            int indexToInsert;

            while ((index + offset) < length) {

                if (array[index + offset] != -1) {
                    indexToInsert = atomicAdd(duplicateCounter, 1);
                    removedArray[indexToInsert] = array[index + offset];
                }

                offset += stride;
            }
        }


        real Launch::collectValues(integer *indices, real *entries, real *collector, integer count) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::collectValues, indices, entries,
                                collector, count);
        }

        template<typename T>
        real Launch::findDuplicates(T *array, integer *duplicateCounter, int length) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::findDuplicates<T>, array,
                                duplicateCounter, length);
        }
        // explicit instantiation for type "integer"
        template real Launch::findDuplicates<integer>(integer *array, integer *duplicateCounter, int length);
        // explicit instantiation for type "real"
        template real Launch::findDuplicates<real>(real *array, integer *duplicateCounter, int length);

        template<typename T>
        real Launch::markDuplicates(T *array, integer *duplicateCounter, int length) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::markDuplicates<T>, array,
                                duplicateCounter, length);
        }
        // explicit instantiation for type "integer"
        template real Launch::markDuplicates<integer>(integer *array, integer *duplicateCounter, int length);
        // explicit instantiation for type "real"
        template real Launch::markDuplicates<real>(real *array, integer *duplicateCounter, int length);

        template<typename T>
        real Launch::removeDuplicates(T *array, T *removedArray, integer *duplicateCounter, int length) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::removeDuplicates<T>, array, removedArray,
                                duplicateCounter, length);
        }
        // explicit instantiation for type "integer"
        template real Launch::removeDuplicates<integer>(integer *array, integer *removedArray,
                integer *duplicateCounter, int length);
        // explicit instantiation for type "real"
        template real Launch::removeDuplicates<real>(real *array, real *removedArray,
                integer *duplicateCounter, int length);
    }

}