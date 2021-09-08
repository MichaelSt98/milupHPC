#ifndef MILUPHPC_CUDA_RUNTIME_H
#define MILUPHPC_CUDA_RUNTIME_H

#include "cuda_utilities.cuh"
#include "../parameter.h"

namespace cuda {

    template <typename T>
    void copy(T *h_var, T *d_var, std::size_t count = 1, To::Target copyTo = To::device) {

        switch (copyTo) {
            case To::device: {
                gpuErrorcheck(cudaMemcpy(d_var, h_var, count * sizeof(T), cudaMemcpyHostToDevice));
            } break;
            case To::host: {
                gpuErrorcheck(cudaMemcpy(h_var, d_var, count * sizeof(T), cudaMemcpyDeviceToHost));
            } break;
            default: {
                printf("cuda::copy Target not available!\n");
            }
        }
    }

    template <typename T>
    void set(T *d_var, T val, std::size_t count = 1) {
        gpuErrorcheck(cudaMemset(d_var, val, count * sizeof(T)));
    }

    template <typename T>
    void malloc(T *&d_var, std::size_t count) {
        gpuErrorcheck(cudaMalloc((void**)&d_var, count * sizeof(T)));
    }

    template <typename T>
    void free(T *d_var) {
        gpuErrorcheck(cudaFree(d_var));
    }


}


#endif //MILUPHPC_CUDA_RUNTIME_H
