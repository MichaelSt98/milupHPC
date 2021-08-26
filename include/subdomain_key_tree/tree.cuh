#ifndef MILUPHPC_TREE_CUH
#define MILUPHPC_TREE_CUH

#include "../cuda_utils/cuda_utilities.cuh"
//#include "../cuda_utils/cuda_launcher.cuh"
#include "../parameter.h"
#include "../particles.cuh"
//#include "subdomain.cuh"

#include <iostream>
#include <stdio.h>
#include <cuda.h>

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

    CUDA_CALLABLE_MEMBER keyType getParticleKey(Particles *particles, integer index, integer maxLevel);

    CUDA_CALLABLE_MEMBER integer getTreeLevel(Particles *particles, integer index, integer maxLevel);

    CUDA_CALLABLE_MEMBER ~Tree();


};

namespace TreeNS {

    namespace Kernel {

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                            integer *index, real *minX, real *maxX);

        namespace Launch {
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted, integer *index,
                     real *minX, real *maxX);
        }

#if DIM > 1

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                            integer *index, real *minX, real *maxX,
                            real *minY, real *maxY);

        namespace Launch {
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                     integer *index, real *minX, real *maxX, real *minY, real *maxY);
        }

#if DIM == 3

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                            integer *index, real *minX, real *maxX, real *minY, real *maxY, real *minZ,
                            real *maxZ);

        namespace Launch {
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                     integer *index, real *minX, real *maxX, real *minY, real *maxY, real *minZ,
                     real *maxZ);
        }

#endif
#endif


        __global__ void buildTree(Tree *tree, Particles *particles, integer n, integer m);

        __global__ void computeBoundingBox(Tree *tree, Particles *particles, integer *mutex,
                                           integer n, integer blockSize);

        __global__ void centerOfMass(Tree *tree, Particles *particles, integer n);

        __global__ void sort(Tree *tree, integer n, integer m);


        __global__ void getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n);

        namespace Launch {

            real getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n,
                                 bool time=false);

            real buildTree(Tree *tree, Particles *particles, integer n, integer m, bool time=false);

            real computeBoundingBox(Tree *tree, Particles *particles, integer *mutex, integer n,
                                          integer blockSize, bool time=false);

            real centerOfMass(Tree *tree, Particles *particles, integer n, bool time=false);

            real sort(Tree *tree, integer n, integer m, bool time=false);

        }

    }
}

#endif //MILUPHPC_TREE_CUH
