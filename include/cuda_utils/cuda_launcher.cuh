/**
 * @file cuda_launcher.cuh
 * @brief CUDA Kernel wrapper execution.
 *
 * Launch CUDA kernels not via `kernel<<<x, y, z>>>(...)` but via
 *
 * ```cpp
 * bool timeKernel {true};
 * real executionTime = 0.;
 * ExecutionPolicy executionPolicy(x, y, z);
 * executionTime = cuda::launch(true, executionPolicy, kernel, ...);
 * ```
 *
 * @author Michael Staneker
 * @bug no known bugs
 * @todo Maximize/optimize occupancy for each kernel.
 */
#ifndef MILUPHPC_CUDALAUNCHER_CUH
#define MILUPHPC_CUDALAUNCHER_CUH

#include "../parameter.h"
#include "cuda_utilities.cuh"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>


/**
 * @brief Execution policy/instruction for CUDA kernel execution.
 */
class ExecutionPolicy {

public:
    //const dim3 gridSize;
    //const dim3 blockSize;
    //const size_t sharedMemBytes;
    /// grid size
    dim3 gridSize;
    /// block size
    dim3 blockSize;
    /// shared memory (bytes)
    size_t sharedMemBytes;
    //const blockSizeInt = ;

    ExecutionPolicy();

    /**
     * @brief Automatically calculate *best* execution policy.
     *
     * Calculate *best* execution policy in means of maximizing occupancy.
     *
     * @warning Currently test version!
     *
     * @tparam Arguments CUDA kernel arguments
     * @param n Particle number to be iterated or more general outer SIMD iteration number.
     * @param f CUDA kernel (function pointer)
     * @param args Arguments of CUDA kernel
     */
    template <typename... Arguments>
    ExecutionPolicy::ExecutionPolicy(int n, void(*f)(Arguments...), Arguments ...args) : sharedMemBytes(0) {
        int _blockSize;
        int minGridSize;
        int _gridSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &_blockSize, *f, 0, 0);
        //cudaOccMaxPotentialOccupancyBlockSize(&minGridSize, &_blockSize, *f, 0, 0);
        // not really beneficial
        //cudaDeviceProp deviceProp;
        //cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
        //int numBlocks;
        //cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &numBlocks, *f, _blockSize, 0 );
        //std::cout << "potential gridSize: " << deviceProp.multiProcessorCount * numBlocks << std::endl;
        //std::cout << deviceProp.multiProcessorCount;
        //end: not really beneficial
        //_blockSize = _blockSize - (_blockSize % 32);
        _gridSize = (n + _blockSize - 1) / _blockSize; //(n/_blockSize) - ((n/_blockSize) % 32); //_blockSize - (_blockSize % 32); //deviceProp.multiProcessorCount * numBlocks; //(n + _blockSize - 1) / _blockSize; // dim3 gridDim(# of SMs in the device * maxActiveBlocks); ?
        blockSize = dim3(_blockSize);
        gridSize = dim3(_gridSize);
        printf("blockSize: %i, gridSize: %i\n", _gridSize, _blockSize);
    }

    /**
     * @brief Constructor for manually setting grid and block size as well as shared memory bytes.
     *
     * @param _gridSize grid size
     * @param _blockSize block size
     * @param _sharedMemBytes shared memory bytes
     */
    ExecutionPolicy(dim3 _gridSize, dim3 _blockSize, size_t _sharedMemBytes);

    /**
     * @brief Constructor for manually setting grid and block size as well as shared memory bytes.
     *
     * @param _gridSize grid size
     * @param _blockSize block size
     */
    ExecutionPolicy(dim3 _gridSize, dim3 _blockSize);
};

namespace cuda {

    /**
     * @brief CUDA execution wrapper function.
     *
     * @tparam Arguments CUDA kernel arguments
     * @param timeKernel time execution of kernel
     * @param policy Execution policy
     * @param f CUDA kernel (function pointer)
     * @param args CUDA kernel actual arguments
     * @return execution time
     */
    template<typename... Arguments>
    real launch(bool timeKernel, const ExecutionPolicy &policy,
                    void (*f)(Arguments...),
                    Arguments... args) {
        float elapsedTime = 0.f;
        ExecutionPolicy p = policy;
        //checkCuda(configureGrid(p, f));
        if (timeKernel) {
            cudaEvent_t start_t, stop_t;
            cudaEventCreate(&start_t);
            cudaEventCreate(&stop_t);
            cudaEventRecord(start_t, 0);

            f<<<p.gridSize, p.blockSize, p.sharedMemBytes>>>(args...);

            cudaEventRecord(stop_t, 0);
            cudaEventSynchronize(stop_t);
            cudaEventElapsedTime(&elapsedTime, start_t, stop_t);
            cudaEventDestroy(start_t);
            cudaEventDestroy(stop_t);
        } else {
            f<<<p.gridSize, p.blockSize, p.sharedMemBytes>>>(args...);
        }

        gpuErrorcheck( cudaPeekAtLastError() );
        gpuErrorcheck( cudaDeviceSynchronize() );

        return elapsedTime;
    }

    /**
     * @brief CUDA execution wrapper function.
     *
     * @tparam Arguments CUDA kernel arguments
     * @param timeKernel time execution of kernel
     * @param f CUDA kernel (function pointer)
     * @param args CUDA kernel actual arguments
     * @return execution time
     */
    template<typename... Arguments>
    real launch(bool timeKernel, void(*f)(Arguments... args), Arguments... args) {
        cudaLaunch(ExecutionPolicy(), f, args...);
    }

}

//#elseif

#endif //MILUPHPC_CUDALAUNCHER_CUH
