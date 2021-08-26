#include "../include/helper_handler.h"

HelperHandler::HelperHandler(integer length) : length(length) {

    gpuErrorcheck(cudaMalloc((void**)&d_integerBuffer, length * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_realBuffer, length * sizeof(real)));

    gpuErrorcheck(cudaMalloc((void**)&d_helper, sizeof(Helper)));
    HelperNS::Kernel::Launch::set(d_helper, d_integerBuffer, d_realBuffer);

}

HelperHandler::~HelperHandler() {

    gpuErrorcheck(cudaFree(d_integerBuffer));
    gpuErrorcheck(cudaFree(d_realBuffer));
    gpuErrorcheck(cudaFree(d_helper));

}