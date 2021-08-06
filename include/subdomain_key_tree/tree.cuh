#ifndef MILUPHPC_TREE_CUH
#define MILUPHPC_TREE_CUH

#include "../cuda_utils/cuda_utilities.cuh"
//#include "../cuda_utils/cuda_launcher.cuh"
#include "../parameter.h"

#include <iostream>
#include <stdio.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

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

/*__global__ void buildTreeKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m)*/

#endif //MILUPHPC_TREE_CUH
