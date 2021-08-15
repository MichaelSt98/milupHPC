//
// Created by Michael Staneker on 15.08.21.
//

#include "../include/helper.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER Helper::Helper() {

}

CUDA_CALLABLE_MEMBER Helper::Helper(integer *intBuffer, real *floatBuffer) : intBuffer(intBuffer),
                floatBuffer(floatBuffer) {

}

CUDA_CALLABLE_MEMBER Helper::~Helper() {

}

CUDA_CALLABLE_MEMBER void Helper::set(integer *intBuffer, real *floatBuffer) {
    this->intBuffer = intBuffer;
    this->floatBuffer = floatBuffer;
}

namespace HelperNS {

    __global__ void setKernel(Helper *helper, integer *intBuffer, real *floatBuffer) {
        helper->set(intBuffer, floatBuffer);
    }

    void launchSetKernel(Helper *helper, integer *intBuffer, real *floatBuffer) {
        ExecutionPolicy executionPolicy(1, 1);
        cuda::launch(false, executionPolicy, setKernel, helper, intBuffer, floatBuffer);
    }
}