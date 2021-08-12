//
// Created by Michael Staneker on 05.08.21.
//

#ifndef MILUPHPC_CUDAUTILITIES_CUH
#define MILUPHPC_CUDAUTILITIES_CUH

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


#define safeCudaCall(call) checkCudaCall(call, #call, __FILE__, __LINE__)
#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void checkCudaCall(cudaError_t command, const char * commandName, const char * fileName, int line);

#endif //MILUPHPC_CUDAUTILITIES_CUH
