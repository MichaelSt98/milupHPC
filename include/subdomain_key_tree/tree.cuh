#ifndef MILUPHPC_TREE_CUH
#define MILUPHPC_TREE_CUH

#include "../cuda_utils/cuda_utilities.cuh"
//#include "../cuda_utils/cuda_launcher.cuh"
#include "../parameter.h"
#include "../particles.cuh"

#include <iostream>
#include <stdio.h>

class Tree {

public:

    integer *count;
    integer *start;
    integer *child;
    integer *sorted;
    integer *index;

    real *minX, *maxX;
#if DIM > 1
    real *minY, *maxY;
#if DIM == 3
    real *minZ, *maxZ;
#endif
#endif

    CUDA_CALLABLE_MEMBER Tree();

    CUDA_CALLABLE_MEMBER Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                              real *minX, real *maxX);
    CUDA_CALLABLE_MEMBER void set(integer *count, integer *start, integer *child, integer *sorted,
                                      integer *index, real *minX, real *maxX);

#if DIM > 1
    CUDA_CALLABLE_MEMBER Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                              real *minX, real *maxX, real *minY, real *maxY);
    CUDA_CALLABLE_MEMBER void set(integer *count, integer *start, integer *child, integer *sorted,
                                      integer *index, real *minX, real *maxX, real *minY, real *maxY);

#if DIM == 3
    CUDA_CALLABLE_MEMBER Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                              real *minX, real *maxX, real *minY, real *maxY, real *minZ, real *maxZ);
    CUDA_CALLABLE_MEMBER void set(integer *count, integer *start, integer *child, integer *sorted,
                                      integer *index, real *minX, real *maxX, real *minY, real *maxY,
                                      real *minZ, real *maxZ);
#endif
#endif

    //TODO: reset pointers! (minX, maxX, ...)
    CUDA_CALLABLE_MEMBER void reset(integer index, integer n);

    CUDA_CALLABLE_MEMBER ~Tree();

};

namespace TreeNS {

    __global__ void setKernel(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                              integer *index, real *minX, real *maxX);

    void launchSetKernel(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                         integer *index, real *minX, real *maxX);

#if DIM > 1
    __global__ void setKernel(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                              integer *index, real *minX, real *maxX,
                              real *minY, real *maxY);

    void launchSetKernel(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                         integer *index, real *minX, real *maxX, real *minY, real *maxY);

#if DIM == 3
    __global__ void setKernel(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                              integer *index, real *minX, real *maxX, real *minY, real *maxY, real *minZ,
                              real *maxZ);

    void launchSetKernel(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                         integer *index, real *minX, real *maxX, real *minY, real *maxY, real *minZ,
                         real *maxZ);
#endif
#endif

    __global__ void buildTreeKernel(Tree *tree, Particles *particles, integer n, integer m);
    void launchBuildTreeKernel(Tree *tree, Particles *particles, integer n, integer m);

    __global__ void computeBoundingBoxKernel(Tree *tree, Particles *particles, integer *mutex, integer n, integer blockSize);
    void launchComputeBoundingBoxKernel(Tree *tree, Particles *particles, integer *mutex, integer n, integer blockSize);

    __global__ void centerOfMassKernel(Tree *tree, Particles *particles, integer n);
    void launchCenterOfMassKernel(Tree *tree, Particles *particles, integer n);

    __global__ void sortKernel(Tree *tree, integer n, integer m);
    void launchSortKernel(Tree *tree, integer n, integer m);
}

#endif //MILUPHPC_TREE_CUH
