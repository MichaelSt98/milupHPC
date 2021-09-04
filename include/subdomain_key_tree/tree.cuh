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

namespace KeyNS {

    //TODO: 1D and 2D DirTable
    // table needed to convert from Lebesgue to Hilbert keys
    CUDA_CALLABLE_MEMBER const unsigned char DirTable[12][8] =
            {{8,  10, 3,  3,  4,  5,  4,  5},
             {2,  2,  11, 9,  4,  5,  4,  5},
             {7,  6,  7,  6,  8,  10, 1,  1},
             {7,  6,  7,  6,  0,  0,  11, 9},
             {0,  8,  1,  11, 6,  8,  6,  11},
             {10, 0,  9,  1,  10, 7,  9,  7},
             {10, 4,  9,  4,  10, 2,  9,  3},
             {5,  8,  5,  11, 2,  8,  3,  11},
             {4,  9,  0,  0,  7,  9,  2,  2},
             {1,  1,  8,  5,  3,  3,  8,  6},
             {11, 5,  0,  0,  11, 6,  2,  2},
             {1,  1,  4,  10, 3,  3,  7,  10}};

    //TODO: 1D and 2d Hilbert table
    // table needed to convert from Lebesgue to Hilbert keys
    CUDA_CALLABLE_MEMBER const unsigned char HilbertTable[12][8] = {{0, 7, 3, 4, 1, 6, 2, 5},
                                                                    {4, 3, 7, 0, 5, 2, 6, 1},
                                                                    {6, 1, 5, 2, 7, 0, 4, 3},
                                                                    {2, 5, 1, 6, 3, 4, 0, 7},
                                                                    {0, 1, 7, 6, 3, 2, 4, 5},
                                                                    {6, 7, 1, 0, 5, 4, 2, 3},
                                                                    {2, 3, 5, 4, 1, 0, 6, 7},
                                                                    {4, 5, 3, 2, 7, 6, 0, 1},
                                                                    {0, 3, 1, 2, 7, 4, 6, 5},
                                                                    {2, 1, 3, 0, 5, 6, 4, 7},
                                                                    {4, 7, 5, 6, 3, 0, 2, 1},
                                                                    {6, 5, 7, 4, 1, 2, 0, 3}};

    CUDA_CALLABLE_MEMBER keyType lebesgue2hilbert(keyType lebesgue, integer maxLevel);

}

class Tree {

public:

    integer *count;
    integer *start;
    integer *child;
    integer *sorted;
    integer *index;

    integer *toDeleteLeaf;
    integer *toDeleteNode;

    real *minX, *maxX;
#if DIM > 1
    real *minY, *maxY;
#if DIM == 3
    real *minZ, *maxZ;
#endif
#endif

    CUDA_CALLABLE_MEMBER Tree();

    CUDA_CALLABLE_MEMBER Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                              integer *toDeleteLeaf, integer *toDeleteNode,
                              real *minX, real *maxX);
    CUDA_CALLABLE_MEMBER void set(integer *count, integer *start, integer *child, integer *sorted,
                                      integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                                      real *minX, real *maxX);

#if DIM > 1
    CUDA_CALLABLE_MEMBER Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                              integer *toDeleteLeaf, integer *toDeleteNode,
                              real *minX, real *maxX, real *minY, real *maxY);
    CUDA_CALLABLE_MEMBER void set(integer *count, integer *start, integer *child, integer *sorted,
                                      integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                                      real *minX, real *maxX, real *minY, real *maxY);

#if DIM == 3
    CUDA_CALLABLE_MEMBER Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                              integer *toDeleteLeaf, integer *toDeleteNode,
                              real *minX, real *maxX, real *minY, real *maxY, real *minZ, real *maxZ);
    CUDA_CALLABLE_MEMBER void set(integer *count, integer *start, integer *child, integer *sorted,
                                      integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                                      real *minX, real *maxX, real *minY, real *maxY,
                                      real *minZ, real *maxZ);
#endif
#endif

    CUDA_CALLABLE_MEMBER void reset(integer index, integer n);

    CUDA_CALLABLE_MEMBER keyType getParticleKey(Particles *particles, integer index, integer maxLevel,
                                                Curve::Type curveType = Curve::lebesgue);

    CUDA_CALLABLE_MEMBER integer getTreeLevel(Particles *particles, integer index, integer maxLevel,
                                              Curve::Type curveType = Curve::lebesgue);

    CUDA_CALLABLE_MEMBER integer sumParticles();

    CUDA_CALLABLE_MEMBER ~Tree();


};

namespace TreeNS {

    namespace Kernel {

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                            integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                            real *minX, real *maxX);

        __global__ void info(Tree *tree, integer n, integer m);

        namespace Launch {
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted, integer *index,
                     integer *toDeleteLeaf, integer *toDeleteNode,
                     real *minX, real *maxX);

            real info(Tree *tree, integer n, integer m);
        }

#if DIM > 1

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                            integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                            real *minX, real *maxX, real *minY, real *maxY);

        namespace Launch {
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                     integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                     real *minX, real *maxX, real *minY, real *maxY);
        }

#if DIM == 3

        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                            integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                            real *minX, real *maxX, real *minY, real *maxY, real *minZ,
                            real *maxZ);

        namespace Launch {
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                     integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                     real *minX, real *maxX, real *minY, real *maxY, real *minZ,
                     real *maxZ);
        }

#endif
#endif

        __global__ void sumParticles(Tree *tree);

        __global__ void buildTree(Tree *tree, Particles *particles, integer n, integer m);

        __global__ void computeBoundingBox(Tree *tree, Particles *particles, integer *mutex,
                                           integer n, integer blockSize);

        __global__ void centerOfMass(Tree *tree, Particles *particles, integer n);

        __global__ void sort(Tree *tree, integer n, integer m);

        __global__ void getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n,
                                        Curve::Type curveType = Curve::lebesgue);

        namespace Launch {

            real sumParticles(Tree *tree);

            real getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n,
                                 Curve::Type curveType = Curve::lebesgue, bool time=false);

            real buildTree(Tree *tree, Particles *particles, integer n, integer m, bool time=false);

            real computeBoundingBox(Tree *tree, Particles *particles, integer *mutex, integer n,
                                          integer blockSize, bool time=false);

            real centerOfMass(Tree *tree, Particles *particles, integer n, bool time=false);

            real sort(Tree *tree, integer n, integer m, bool time=false);

        }

    }
}

#endif //MILUPHPC_TREE_CUH
