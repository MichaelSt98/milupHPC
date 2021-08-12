#ifndef MILUPHPC_TREE_CUH
#define MILUPHPC_TREE_CUH

#include "../cuda_utils/cuda_utilities.cuh"
//#include "../cuda_utils/cuda_launcher.cuh"
#include "../parameter.h"

#include <iostream>
#include <stdio.h>

__global__ void buildTreeKernel(float *x, float *y, float *z, float *mass, int *count, int *start,
                                int *child, int *index, float *minX, float *maxX, float *minY, float *maxY,
                                float *minZ, float *maxZ, int n, int m);

#endif //MILUPHPC_TREE_CUH
