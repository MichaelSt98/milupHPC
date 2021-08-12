#ifndef MILUPHPC_SAMPLE_CUH
#define MILUPHPC_SAMPLE_CUH

#include "../cuda_utils/cuda_utilities.cuh"
//#include "../cuda_utils/cuda_launcher.cuh"
#include "../parameter.h"

#include <iostream>
#include <stdio.h>

class Foo {
public:

    int *d_test;

    CUDA_CALLABLE_MEMBER Foo();
    CUDA_CALLABLE_MEMBER Foo(int *test);
    CUDA_CALLABLE_MEMBER ~Foo();
    CUDA_CALLABLE_MEMBER void aMethod(int *test);

};

__global__ void setKernel(Foo *foo, int *test);
__global__ void testKernel(Foo *foo);

void launchSetKernel(Foo *foo, int *test);
void launchTestKernel(Foo *foo);

#endif //MILUPHPC_TREE_CUH
