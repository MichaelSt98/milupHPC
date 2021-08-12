#include "../../include/subdomain_key_tree/sample.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER Foo::Foo() {

}

CUDA_CALLABLE_MEMBER Foo::Foo(int *test) : d_test(test) {

}

CUDA_CALLABLE_MEMBER Foo::~Foo() {

}

CUDA_CALLABLE_MEMBER void Foo::aMethod(int *test) {
    d_test = test;
}

__global__ void setKernel(Foo *foo, int *test) {
    //foo->d_test = test;
    foo->aMethod(test);
}

__global__ void testKernel(Foo *foo) {

    for (int i=0; i<5; i++) {
        foo->d_test[i] = i;
        printf("<<<testKernel>>> test = %i\n", foo->d_test[i]);
    }

}

void launchTestKernel(Foo *foo) {
    ExecutionPolicy executionPolicy(1, 1);
    cudaLaunch(false, executionPolicy, testKernel, foo);
    //testKernel<<<1, 1>>>(foo);
}

void launchSetKernel(Foo *foo, int *test) {
    setKernel<<<1, 1>>>(foo, test);
}