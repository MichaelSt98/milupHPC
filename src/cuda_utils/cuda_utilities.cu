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

        __global__ void checkValues(integer *indices, real *entry1, real *entry2, real *entry3,
                                      integer count) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((index + offset) < count) {
                for (int i=0; i<count; i++) {
                    if (i != index + offset) {
                        if (entry1[indices[i]] == entry1[indices[index + offset]] && entry2[indices[i]] == entry2[indices[index + offset]]) {
                            printf("Same entry for %i vs %i | %i vs %i!\n", i, index + offset, indices[i], indices[index + offset]);
                        }
                    }
                }
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
                    if (i != (index + offset)) {
                        if (array[index + offset] == array[i]) {
                            atomicAdd(duplicateCounter, 1);
                        }
                    }
                }
                offset += stride;
            }
        }

        template<typename T>
        __global__ void findDuplicates(T *array1, T *array2, integer *duplicateCounter, int length) {
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((bodyIndex + offset) < length) {

                for (int i=0; i<length; i++) {
                    if (i != (bodyIndex + offset)) {
                        if (array1[bodyIndex + offset] == array1[i] && array2[bodyIndex + offset] == array2[i]) {
                            atomicAdd(duplicateCounter, 1);
                            printf("duplicate! (%i vs. %i) (x = %f, y = %f)\n", i, bodyIndex + offset, array1[i], array2[i]);
                        }
                    }
                }

                offset += stride;
            }
        }

        template<typename T, typename U>
        __global__ void findDuplicates(T *array, U *entry1, U *entry2, integer *duplicateCounter, int length) {
            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            while ((index + offset) < length) {
                if ((index + offset) % 100 == 0) {
                    //printf("array[%i] = %i\n", index+offset, array[index + offset]);
                }
                for (int i=0; i<length; i++) {
                    if (i != (index + offset)) {
                        if (array[index + offset] == array[i] || (entry1[array[index + offset]] == entry1[array[i]] && entry2[array[index + offset]] == entry2[array[i]])) {
                        //if (array[index + offset] == array[i]) { //|| (entry1[index + offset] = entry1[i] && entry2[index + offset] == entry2[i])) {
                            printf("Found duplicate!\n");
                            atomicAdd(duplicateCounter, 1);
                        }
                    }
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

        template<typename T, typename U>
        __global__ void markDuplicates(T *array, U *entry1, U *entry2, integer *duplicateCounter, int length) {

            integer index = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;
            integer maxIndex;

            //remark: check only x, but in principle check all
            while ((index + offset) < length) {
                if (array[index + offset] != -1) {
                    for (integer i = 0; i < length; i++) {
                        if (i != (index + offset)) {
                            if (array[i] != -1 && (array[index + offset] == array[i] || (entry1[array[i]] == entry1[array[index + offset]] &&
                                    entry2[array[i]] == entry2[array[index + offset]]))) {
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

        real Launch::checkValues(integer *indices, real *entry1, real *entry2, real *entry3, integer count) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::checkValues, indices, entry1, entry2, entry3,
                                count);
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
        real Launch::findDuplicates(T *array1, T *array2, integer *duplicateCounter, int length) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::findDuplicates<T>, array1,
                                array2, duplicateCounter, length);
        }
        //template real Launch::findDuplicates<real>(real *array1, real *array2, integer *duplicateCounter, int length);

        template<typename T, typename U>
        real Launch::findDuplicates(T *array, U *entry1, U *entry2, integer *duplicateCounter, int length) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::findDuplicates<T, U>, array, entry1, entry2,
                                duplicateCounter, length);
        }
        template real Launch::findDuplicates<integer, real>(integer *array, real *entry1, real *entry2,
                integer *duplicateCounter, int length);

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

        template<typename T, typename U>
        real Launch::markDuplicates(T *array, U *entry1, U *entry2, integer *duplicateCounter, int length) {
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::CudaUtils::Kernel::markDuplicates<T, U>, array, entry1, entry2,
                                duplicateCounter, length);
        }
        template real Launch::markDuplicates<integer, real>(integer *array, real *entry1, real *entry2, integer *duplicateCounter, int length);

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