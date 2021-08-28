#include "../include/helper_handler.h"

HelperHandler::HelperHandler(integer length) : length(length) {

    gpuErrorcheck(cudaMalloc((void**)&d_integerVal, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_realVal, sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_keyTypeVal, sizeof(keyType)));

    gpuErrorcheck(cudaMalloc((void**)&d_integerBuffer, length * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_realBuffer, length * sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_keyTypeBuffer, length * sizeof(keyType)));

    gpuErrorcheck(cudaMalloc((void**)&d_helper, sizeof(Helper)));
    HelperNS::Kernel::Launch::set(d_helper, d_integerVal, d_realVal, d_keyTypeVal, d_integerBuffer,
                                  d_realBuffer, d_keyTypeBuffer);

}

HelperHandler::~HelperHandler() {

    gpuErrorcheck(cudaFree(d_integerVal));
    gpuErrorcheck(cudaFree(d_realVal));
    gpuErrorcheck(cudaFree(d_keyTypeVal));

    gpuErrorcheck(cudaFree(d_integerBuffer));
    gpuErrorcheck(cudaFree(d_realBuffer));
    gpuErrorcheck(cudaFree(d_keyTypeBuffer));

    gpuErrorcheck(cudaFree(d_helper));

}

void HelperHandler::reset() {
    gpuErrorcheck(cudaMemset(d_integerVal, 0, sizeof(integer)));
    gpuErrorcheck(cudaMemset(d_realVal, 0., sizeof(real)));
    gpuErrorcheck(cudaMemset(d_keyTypeVal, 0, sizeof(keyType)));

    gpuErrorcheck(cudaMemset(d_integerBuffer, 0, length * sizeof(integer)));
    gpuErrorcheck(cudaMemset(d_realBuffer, 0., length * sizeof(real)));
    gpuErrorcheck(cudaMemset(d_keyTypeBuffer, 0, length * sizeof(keyType)));
}