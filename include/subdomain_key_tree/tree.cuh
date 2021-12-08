#ifndef MILUPHPC_TREE_CUH
#define MILUPHPC_TREE_CUH

#include "../cuda_utils/cuda_utilities.cuh"
#include "../parameter.h"
#include "../particles.cuh"

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cmath>

//TODO: compiling on binac:
// src/gravity/../../include/gravity/../subdomain_key_tree/tree.cuh(26): error: attribute "__host__" does not apply here
// src/gravity/../../include/gravity/../subdomain_key_tree/tree.cuh(53): error: attribute "__host__" does not apply here
namespace KeyNS {

    /**
 * Table needed to convert from Lebesgue to Hilbert keys
 */
#if DIM == 1
    CUDA_CALLABLE_MEMBER const unsigned char DirTable[1][1] = {{1}}; //TODO: 1D DirTable?
#elif DIM == 2
    CUDA_CALLABLE_MEMBER const unsigned char DirTable[4][4] =
            {{1,2,0,0},
             {0,1,3,1},
             {2,0,2,3},
             {3,3,1,2}};
#else DIM == 3
#ifndef __CUDACC__
    const unsigned char DirTable[12][8] =
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
#else
    __device__ const unsigned char DirTable[12][8] =
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
#endif
#endif

    /**
     * Table needed to convert from Lebesgue to Hilbert keys
     */
#if DIM == 1
    CUDA_CALLABLE_MEMBER const unsigned char HilbertTable[1][1] = {{1}}; //TODO: 1D HilbertTable?
#elif DIM == 2
    CUDA_CALLABLE_MEMBER const unsigned char HilbertTable[4][4] =
            {{0,3,1,2},
             {0,1,3,2},
             {2,3,1,0},
             {2,1,3,0}};
#else
#ifndef __CUDACC__
    const unsigned char HilbertTable[12][8] = {{0, 7, 3, 4, 1, 6, 2, 5},
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
#else
    __device__ const unsigned char HilbertTable[12][8] = {{0, 7, 3, 4, 1, 6, 2, 5},
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
#endif
#endif

    /**
     * Convert a Lebesgue key to a Hilbert key
     *
     * @param lebesgue Lebesgue key
     * @param maxLevel Maximum tree level
     * @return Hilbert key
     */
    CUDA_CALLABLE_MEMBER keyType lebesgue2hilbert(keyType lebesgue, integer maxLevel);

    CUDA_CALLABLE_MEMBER keyType lebesgue2hilbert(keyType lebesgue, int maxLevel, int level);

}

/**
 * Tree class.
 *
 * Class to build and store hierarchical tree structure.
 */
class Tree {

public:

    //TODO: count, start, sorted currently unused (since sort() and computeForces() described by Griebel not used!)
    /// accumulated nodes/leaves beneath
    integer *count;
    /// TODO: describe start
    integer *start;
    /// children/child nodes or leaves (beneath)
    integer *child;
    /// sorted (indices) for better cache performance
    integer *sorted;
    /// index used for counting nodes
    integer *index;

    /// buffer for remembering old indices for rebuilding tree
    integer *toDeleteLeaf;
    /// buffer for remembering old indices for rebuilding tree
    integer *toDeleteNode;

    /// bounding box minimal x
    real *minX;
    /// bounding box maximal x
    real *maxX;
#if DIM > 1
    /// bounding box minimal y
    real *minY;
    /// bounding box maximal y
    real *maxY;
#if DIM == 3
    /// bounding box minimal z
    real *minZ;
    /// bounding box maximal z
    real *maxZ;
#endif
#endif

    /**
     * Default constructor
     */
    CUDA_CALLABLE_MEMBER Tree();

    /**
     * Constructor, passing pointer to member variables
     *
     * @param count allocated array for accumulated nodes/leaves beneath
     * @param start allocated array for
     * @param child allocated array for children
     * @param sorted allocated array for sorted (indices) for better cache performance
     * @param index allocated value for index used for counting nodes
     * @param toDeleteLeaf allocated array for remembering old indices for rebuilding tree
     * @param toDeleteNode allocated array for remembering old indices for rebuilding tree
     * @param minX allocated value for bounding box minimal x
     * @param maxX allocated value for bounding box maximal x
     */
    CUDA_CALLABLE_MEMBER Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                              integer *toDeleteLeaf, integer *toDeleteNode,
                              real *minX, real *maxX);
    /**
     * Setter, passing pointer to member variables
     *
     * @param count allocated array for accumulated nodes/leaves beneath
     * @param start allocated array for
     * @param child allocated array for children
     * @param sorted allocated array for sorted (indices) for better cache performance
     * @param index allocated value for index used for counting nodes
     * @param toDeleteLeaf allocated array for remembering old indices for rebuilding tree
     * @param toDeleteNode allocated array for remembering old indices for rebuilding tree
     * @param minX allocated value for bounding box minimal x
     * @param maxX allocated value for bounding box maximal x
     */
    CUDA_CALLABLE_MEMBER void set(integer *count, integer *start, integer *child, integer *sorted,
                                      integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                                      real *minX, real *maxX);

#if DIM > 1
    /**
     * Constructor, passing pointer to member variables
     *
     * @param count allocated array for accumulated nodes/leaves beneath
     * @param start allocated array for
     * @param child allocated array for children
     * @param sorted allocated array for sorted (indices) for better cache performance
     * @param index allocated value for index used for counting nodes
     * @param toDeleteLeaf allocated array for remembering old indices for rebuilding tree
     * @param toDeleteNode allocated array for remembering old indices for rebuilding tree
     * @param minX allocated value for bounding box minimal x
     * @param maxX allocated value for bounding box maximal x
     * @param minY allocated value for bounding box minimal y
     * @param maxY allocated value for bounding box maximal y
     */
    CUDA_CALLABLE_MEMBER Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                              integer *toDeleteLeaf, integer *toDeleteNode,
                              real *minX, real *maxX, real *minY, real *maxY);
    /**
     * Setter, passing pointer to member variables
     *
     * @param count allocated array for accumulated nodes/leaves beneath
     * @param start allocated array for
     * @param child allocated array for children
     * @param sorted allocated array for sorted (indices) for better cache performance
     * @param index allocated value for index used for counting nodes
     * @param toDeleteLeaf allocated array for remembering old indices for rebuilding tree
     * @param toDeleteNode allocated array for remembering old indices for rebuilding tree
     * @param minX allocated value for bounding box minimal x
     * @param maxX allocated value for bounding box maximal x
     * @param minY allocated value for bounding box minimal y
     * @param maxY allocated value for bounding box maximal y
     */
    CUDA_CALLABLE_MEMBER void set(integer *count, integer *start, integer *child, integer *sorted,
                                      integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                                      real *minX, real *maxX, real *minY, real *maxY);

#if DIM == 3
    /**
     * Constructor, passing pointer to member variables
     *
     * @param count allocated array for accumulated nodes/leaves beneath
     * @param start allocated array for
     * @param child allocated array for children
     * @param sorted allocated array for sorted (indices) for better cache performance
     * @param index allocated value for index used for counting nodes
     * @param toDeleteLeaf allocated array for remembering old indices for rebuilding tree
     * @param toDeleteNode allocated array for remembering old indices for rebuilding tree
     * @param minX allocated value for bounding box minimal x
     * @param maxX allocated value for bounding box maximal x
     * @param minY allocated value for bounding box minimal y
     * @param maxY allocated value for bounding box maximal y
     * @param minZ allocated value for bounding box minimal z
     * @param maxZ allocated value for bounding box maximal z
     */
    CUDA_CALLABLE_MEMBER Tree(integer *count, integer *start, integer *child, integer *sorted, integer *index,
                              integer *toDeleteLeaf, integer *toDeleteNode,
                              real *minX, real *maxX, real *minY, real *maxY, real *minZ, real *maxZ);
    /**
     * Constructor, passing pointer to member variables
     *
     * @param count allocated array for accumulated nodes/leaves beneath
     * @param start allocated array for
     * @param child allocated array for children
     * @param sorted allocated array for sorted (indices) for better cache performance
     * @param index allocated value for index used for counting nodes
     * @param toDeleteLeaf allocated array for remembering old indices for rebuilding tree
     * @param toDeleteNode allocated array for remembering old indices for rebuilding tree
     * @param minX allocated value for bounding box minimal x
     * @param maxX allocated value for bounding box maximal x
     * @param minY allocated value for bounding box minimal y
     * @param maxY allocated value for bounding box maximal y
     * @param minZ allocated value for bounding box minimal z
     * @param maxZ allocated value for bounding box maximal z
     */
    CUDA_CALLABLE_MEMBER void set(integer *count, integer *start, integer *child, integer *sorted,
                                      integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                                      real *minX, real *maxX, real *minY, real *maxY,
                                      real *minZ, real *maxZ);
#endif
#endif

    /**
     * Reset (specific) entries
     *
     * @param index
     * @param n
     */
    CUDA_CALLABLE_MEMBER void reset(integer index, integer n);

    /**
     *
     * @param particles instance of particles (array)
     * @param index desired index in particles to get tree of desired particle
     * @param maxLevel max tree level
     * @param curveType space-filling curve type to be assumed (Lebesgue/Hilbert)
     * @return particle key
     */
    CUDA_CALLABLE_MEMBER keyType getParticleKey(Particles *particles, integer index, integer maxLevel,
                                                Curve::Type curveType = Curve::lebesgue);

    /**
     * Get tree level for a (specific) particle.
     *
     * Calculates the particle key and uses key to construct path within tree, returning when path lead to
     * the desired particle/index of the particle.
     *
     * @param particles particles instance of particles (array)
     * @param index desired index in particles to get tree of desired particle
     * @param maxLevel max tree level
     * @param curveType space-filling curve type to be assumed (Lebesgue/Hilbert)
     * @return particles[index] tree level
     */
    CUDA_CALLABLE_MEMBER integer getTreeLevel(Particles *particles, integer index, integer maxLevel,
                                              Curve::Type curveType = Curve::lebesgue);

    /**
     * Sum particles in tree.
     *
     * @return sum of particles within tree
     */
    CUDA_CALLABLE_MEMBER integer sumParticles();

    /**
     * Destructor
     */
    CUDA_CALLABLE_MEMBER ~Tree();


};

namespace TreeNS {

    namespace Kernel {

        /**
         * Kernel call to setter
         *
         * @param tree
         * @param count
         * @param start
         * @param child
         * @param sorted
         * @param index
         * @param toDeleteLeaf
         * @param toDeleteNode
         * @param minX
         * @param maxX
         */
        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                            integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                            real *minX, real *maxX);

        /**
         * Info Kernel for tree class (for debugging purposes)
         *
         * @param tree
         * @param n
         * @param m
         */
        __global__ void info(Tree *tree, Particles *particles, integer n, integer m);

        __global__ void info(Tree *tree, Particles *particles);

        __global__ void testTree(Tree *tree, Particles *particles, integer n, integer m);

        namespace Launch {
            /**
             * Wrapped kernel call to setter
             *
             * @param tree
             * @param count
             * @param start
             * @param child
             * @param sorted
             * @param index
             * @param toDeleteLeaf
             * @param toDeleteNode
             * @param minX
             * @param maxX
             */
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted, integer *index,
                     integer *toDeleteLeaf, integer *toDeleteNode,
                     real *minX, real *maxX);

            /**
             * Wrapped kernel for tree class (for debugging purposes)
             */
            real info(Tree *tree, Particles *particles, integer n, integer m);

            real info(Tree *tree, Particles *particles);

            real testTree(Tree *tree, Particles *particles, integer n, integer m);
        }

#if DIM > 1
        /**
         * Kernel call to setter
         *
         * @param tree
         * @param count
         * @param start
         * @param child
         * @param sorted
         * @param index
         * @param toDeleteLeaf
         * @param toDeleteNode
         * @param minX
         * @param maxX
         * @param minY
         * @param maxY
         */
        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                            integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                            real *minX, real *maxX, real *minY, real *maxY);

        namespace Launch {
            /**
             * Wrapped kernel call to setter
             *
             * @param tree
             * @param count
             * @param start
             * @param child
             * @param sorted
             * @param index
             * @param toDeleteLeaf
             * @param toDeleteNode
             * @param minX
             * @param maxX
             * @param minY
             * @param maxY
             */
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                     integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                     real *minX, real *maxX, real *minY, real *maxY);
        }

#if DIM == 3

        /**
         * Kernel call to setter
         *
         * @param tree
         * @param count
         * @param start
         * @param child
         * @param sorted
         * @param index
         * @param toDeleteLeaf
         * @param toDeleteNode
         * @param minX
         * @param maxX
         * @param minY
         * @param maxY
         * @param minZ
         * @param maxZ
         */
        __global__ void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                            integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                            real *minX, real *maxX, real *minY, real *maxY, real *minZ,
                            real *maxZ);

        namespace Launch {
            /**
             * Wrapped kernel call to setter
             *
             * @param tree
             * @param count
             * @param start
             * @param child
             * @param sorted
             * @param index
             * @param toDeleteLeaf
             * @param toDeleteNode
             * @param minX
             * @param maxX
             * @param minY
             * @param maxY
             * @param minZ
             * @param maxZ
             */
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted,
                     integer *index, integer *toDeleteLeaf, integer *toDeleteNode,
                     real *minX, real *maxX, real *minY, real *maxY, real *minZ,
                     real *maxZ);
        }

#endif
#endif

        /**
         * Kernel call to sum particles within tree
         *
         * @param tree target tree instance
         */
        __global__ void sumParticles(Tree *tree);

        /**
         * Kernel to construct the tree using the particles within `particles`
         *
         * @param tree tree target instance
         * @param particles particles to be inserted in tree
         * @param n number of particles
         * @param m number of potential particles to be inserted (needed for start index of nodes)
         */
        __global__ void buildTree(Tree *tree, Particles *particles, integer n, integer m);

        __global__ void prepareSorting(Tree *tree, Particles *particles, integer n, integer m);

        __global__ void calculateCentersOfMass(Tree *tree, Particles *particles, integer n, integer level);

        /**
         * Kernel to compute the bounding box/borders of the tree or rather the particles within the tree
         *
         * @param tree tree target instance
         * @param particles particles within the tree
         * @param mutex mutex/lock
         * @param n number of particles
         * @param blockSize device block size
         */
        __global__ void computeBoundingBox(Tree *tree, Particles *particles, integer *mutex,
                                           integer n, integer blockSize);

        /**
         * Kernel to compute center of mass for pseudo-particles/nodes within tree
         *
         * @param tree
         * @param particles
         * @param n
         */
        __global__ void centerOfMass(Tree *tree, Particles *particles, integer n);

        /**
         * Kernel to sort tree/child indices to optimize cache efficiency
         *
         * @param tree
         * @param n
         * @param m
         */
        __global__ void sort(Tree *tree, integer n, integer m);

        /**
         * Kernel to get all particle's keys
         *
         * @param tree
         * @param particles
         * @param[out] keys input particle's keys
         * @param maxLevel
         * @param n
         * @param curveType
         */
        __global__ void getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n,
                                        Curve::Type curveType = Curve::lebesgue);

        __global__ void globalCOM(Tree *tree, Particles *particles, real com[DIM]);

        namespace Launch {

            /**
             * Wrapped kernel call to sum particles within tree.
             *
             * @param tree
             * @return
             */
            real sumParticles(Tree *tree);

            /**
             * Wrapped kernel call to get all particle's key(s)
             *
             * @param tree
             * @param particles
             * @param keys
             * @param maxLevel
             * @param n
             * @param curveType
             * @param time
             * @return
             */
            real getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n,
                                 Curve::Type curveType = Curve::lebesgue, bool time=false);

            /**
             * Wrapped kernel call to build tree structure
             *
             * @param tree
             * @param particles
             * @param n
             * @param m
             * @param time
             * @return
             */
            real buildTree(Tree *tree, Particles *particles, integer n, integer m, bool time=false);

            real prepareSorting(Tree *tree, Particles *particles, integer n, integer m);

            real calculateCentersOfMass(Tree *tree, Particles *particles, integer n, integer level, bool time=false);

            /**
             * Wrapped kernel call to compute bounding box(es)/borders
             *
             * @param tree
             * @param particles
             * @param mutex
             * @param n
             * @param blockSize
             * @param time
             * @return
             */
            real computeBoundingBox(Tree *tree, Particles *particles, integer *mutex, integer n,
                                          integer blockSize, bool time=false);

            /**
             * Wrapped kernel call to compute center of mass
             *
             * @param tree
             * @param particles
             * @param n
             * @param time
             * @return
             */
            real centerOfMass(Tree *tree, Particles *particles, integer n, bool time=false);

            /**
             * Wrapped kernel call to sort tree (indices)
             *
             * @param tree
             * @param n
             * @param m
             * @param time
             * @return
             */
            real sort(Tree *tree, integer n, integer m, bool time=false);

            real globalCOM(Tree *tree, Particles *particles, real com[DIM]);

        }
    }
}

#endif //MILUPHPC_TREE_CUH
