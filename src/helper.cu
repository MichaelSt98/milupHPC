//
// Created by Michael Staneker on 15.08.21.
//

#include "../include/helper.cuh"
#include "../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER Helper::Helper() {

}

CUDA_CALLABLE_MEMBER Helper::Helper(integer *integerBuffer, real *realBuffer) : integerBuffer(integerBuffer),
                realBuffer(realBuffer) {

}

CUDA_CALLABLE_MEMBER Helper::~Helper() {

}

CUDA_CALLABLE_MEMBER void Helper::set(integer *integerBuffer, real *realBuffer) {
    this->integerBuffer = integerBuffer;
    this->realBuffer = realBuffer;
}

namespace HelperNS {

    namespace Kernel {
        __global__ void set(Helper *helper, integer *integerBuffer, real *realBuffer) {
            helper->set(integerBuffer, realBuffer);
        }

        void Launch::set(Helper *helper, integer *integerBuffer, real *realBuffer) {
            ExecutionPolicy executionPolicy(1, 1);
            cuda::launch(false, executionPolicy, ::HelperNS::Kernel::Launch::set, helper, integerBuffer,
                         realBuffer);

        }
    }
}