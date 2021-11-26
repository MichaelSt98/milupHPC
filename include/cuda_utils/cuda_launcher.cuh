#ifndef MILUPHPC_CUDALAUNCHER_CUH
#define MILUPHPC_CUDALAUNCHER_CUH

#include "../parameter.h"
#include "cuda_utilities.cuh"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>


class ExecutionPolicy {

public:
    //const dim3 gridSize;
    //const dim3 blockSize;
    //const size_t sharedMemBytes;
    dim3 gridSize;
    dim3 blockSize;
    size_t sharedMemBytes;
    //const blockSizeInt = ;

    ExecutionPolicy();

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

    ExecutionPolicy(dim3 _gridSize, dim3 _blockSize, size_t _sharedMemBytes);
    ExecutionPolicy(dim3 _gridSize, dim3 _blockSize);
};

namespace cuda {
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

    template<typename... Arguments>
    real launch(bool timeKernel, void(*f)(Arguments... args), Arguments... args) {
        cudaLaunch(ExecutionPolicy(), f, args...);
    }

}

//#elseif

#endif //MILUPHPC_CUDALAUNCHER_CUH
