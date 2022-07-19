/**
 * @file tree.cuh
 * @brief Tree related classes, kernels and functions.
 *
 *
 * @author Michael Staneker
 * @bug no known bugs
 * @todo: compiling on binac:
 *   src/gravity/../../include/gravity/../subdomain_key_tree/tree.cuh(26): error: attribute "__host__" does not apply here
 *   src/gravity/../../include/gravity/../subdomain_key_tree/tree.cuh(53): error: attribute "__host__" does not apply here
 */
#ifndef MILUPHPC_TREE_CUH
#define MILUPHPC_TREE_CUH

#include "../parameter.h"
#include "../cuda_utils/cuda_utilities.cuh"
#include "../particles.cuh"
#include "../box.h"

#include <iostream>
#include <stdio.h>
#include "utils/logger.h"
#if TARGET_GPU
#include <cuda.h>
#endif
#include <assert.h>
#include <cmath>

namespace KeyNS {

/**
 * @brief Table needed to convert from Lebesgue to Hilbert keys
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
 * @brief Table needed to convert from Lebesgue to Hilbert keys
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
     * @brief Convert a Lebesgue key to a Hilbert key
     *
     * @param lebesgue Lebesgue key
     * @param maxLevel Maximum tree level
     * @return Hilbert key
     */
    CUDA_CALLABLE_MEMBER keyType lebesgue2hilbert(keyType lebesgue, integer maxLevel);

    CUDA_CALLABLE_MEMBER keyType lebesgue2hilbert(keyType lebesgue, int maxLevel, int level);

}

/**
 * @brief Tree class.
 *
 * Class to build and store hierarchical tree structure.
 *
 * Since dynamic memory allocation and access to heap objects in GPUs are usually suboptimal,
 * an array-based data structure is used to store the (pseudo-)particle information as well as the tree
 * and allows for efficient cache alignment. Consequently, array indices are used instead of pointers
 * to constitute the tree out of the tree nodes, whereas "-1" represents a null pointer and "-2" is
 * used for locking nodes. For this purpose an array with the minimum length \f$ 2^{d} \cdot (M - N) \f$
 * with dimensionality \f$ d \f$ and number of cells \f$ (M -N) \f$ is needed to store the children.
 * The \f$ i \f$-th child of node \f$ c \f$ can therefore be accessed via index \f$ 2^d \cdot c + i \f$.
 *
 * \image html images/Parallelization/tree_layout.png width=40%
 *
 */
class Tree {

public:

    //TODO: count, start, sorted currently unused (since sort() and computeForces() described by Griebel not used!)
    /// accumulated nodes/leaves beneath @deprecated
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
     * @brief Default constructor
     */
    CUDA_CALLABLE_MEMBER Tree();

    /**
     * @brief Constructor, passing pointer to member variables
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
     * @brief Setter, passing pointer to member variables
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
     * @brief Constructor, passing pointer to member variables
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
     * @brief Setter, passing pointer to member variables
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
     * @brief Constructor, passing pointer to member variables
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
     * @brief Constructor, passing pointer to member variables
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
     * @brief Reset (specific) entries
     *
     * @param index
     * @param n
     */
    CUDA_CALLABLE_MEMBER void reset(integer index, integer n);

    /**
     * @brief Get SFC key of a particle.
     *
     * The particle key computation can be accomplished by using the particle's location within the simulation domain,
     * thus regarding the bounding boxes, which is equivalent to a tree traversing.
     *
     * The Lebesgue keys can be converted via tables.
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
     * @brief Get tree level for a (specific) particle.
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
     * @brief Sum particles in tree.
     *
     * @return sum of particles within tree
     */
    CUDA_CALLABLE_MEMBER integer sumParticles();

    /**
     * @brief Destructor
     */
    CUDA_CALLABLE_MEMBER ~Tree();

    bool isLeaf(integer nodeIndex);

    bool isDomainList(integer nodeIndex);

    void buildTree(Particles *particles, integer numParticlesLocal, integer numParticles);

    void insertTree(Particles *particles, integer particleIndex, integer nodeIndex, integer numParticles, Box &box);

};

#if TARGET_GPU
namespace TreeNS {

    namespace Kernel {

        /**
         * @brief Kernel call to setter
         *
         * > Corresponding wrapper function: ::TreeNS::Kernel::Launch::set()
         *
         * @param tree Tree class instance (to be constructed)
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
         * @brief Info Kernel for tree class (for debugging purposes)
         *
         * > Corresponding wrapper function: ::TreeNS::Kernel::Launch::info()
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
             * @brief Wrapper for ::TreeNS::Kernel::set()
             */
            void set(Tree *tree, integer *count, integer *start, integer *child, integer *sorted, integer *index,
                     integer *toDeleteLeaf, integer *toDeleteNode,
                     real *minX, real *maxX);

            /**
             * @brief Wrapper for ::TreeNS::Kernel::Launch::info()
             */
            real info(Tree *tree, Particles *particles, integer n, integer m);

            real info(Tree *tree, Particles *particles);

            real testTree(Tree *tree, Particles *particles, integer n, integer m);
        }

#if DIM > 1
        /**
         * @brief Kernel call to setter
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
             * @brief Wrapper for ::TreeNS::Kernel::Launch::set()
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
             * @brief Wrapper for ::TreeNS::Kernel::Launch::set()
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
         * @brief Kernel call to sum particles within tree
         *
         * > Corresponding wrapper function: ::TreeNS::Kernel::Launch::sumParticles()
         *
         * @param tree target tree instance
         */
        __global__ void sumParticles(Tree *tree);

        /**
         * @brief Kernel to construct the tree using the particles within `particles`
         *
         * > Corresponding wrapper function: ::TreeNS::Kernel::Launch::buildTree()
         *
         * The algorithm shows an iterative tree construction algorithm utilizing lightweight locks for leaf nodes
         * to avoid race conditions. Particles are assigned to threads, those threads insert its body by traversing
         * the tree from root to the correct node trying to lock the corresponding child pointer via the array
         * index "-2" using atomic operations. If the locking is successful the particle is inserted into the tree,
         * releasing the lock and executing a memory fence to ensure visibility for the other threads. If a particle
         * is already stored at the desired node, a new cell is automatically generated or rather requested and the
         * old and new body are inserted correspondingly. However, if the locking is not successful, the thread need
         * to try again until accomplishing the task. This approach is very similar to the one presented in
         * [An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf).
         *
         * @param tree Tree class target instance
         * @param particles Particles class instance/particles to be inserted in tree
         * @param n number of particles
         * @param m number of potential particles to be inserted (needed for start index of nodes)
         */
        __global__ void buildTree(Tree *tree, Particles *particles, integer n, integer m);

        __global__ void prepareSorting(Tree *tree, Particles *particles, integer n, integer m);

        /**
         * @brief Compute center of masses (level wise).
         *
         * The local COM computation kernel is called iteratively starting with the deepest tree level and
         * finishing at the root. The pseudo-particles children entries are weighted to derive the COM, their
         * masses are accumulated and finally the position is normalized by dividing through the summed mass.
         *
         * @param tree Tree class instance
         * @param particles Particle class instance
         * @param n Number of particles
         * @param level Current (relevant) tree level
         */
        __global__ void calculateCentersOfMass(Tree *tree, Particles *particles, integer n, integer level);

        /**
         * @brief Kernel to compute the bounding box/borders of the tree or rather the particles within the tree
         *
         * > Corresponding wrapper function: ::TreeNS::Kernel::Launch::computeBoundingBox()
         *
         * The algorithm to compute the bounding box is basically a parallel reduction primitive to derive the
         * minimum and maximum for each coordinate axis.
         *
         * @param tree Tree class target instance
         * @param particles Particles class instance/particles within the tree
         * @param mutex mutex/lock
         * @param n number of particles
         * @param blockSize device block size
         */
        __global__ void computeBoundingBox(Tree *tree, Particles *particles, integer *mutex,
                                           integer n, integer blockSize);

        /**
         * Kernel to compute center of mass for pseudo-particles/nodes within tree
         *
         * > Corresponding wrapper function: ::TreeNS::Kernel::Launch::centerOfMass()
         *
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param n
         */
        __global__ void centerOfMass(Tree *tree, Particles *particles, integer n);

        /**
         * @brief Kernel to sort tree/child indices to optimize cache efficiency
         *
         * > Corresponding wrapper function: ::TreeNS::Kernel::Launch::sort()
         *
         * @param tree Tree class instance
         * @param n
         * @param m
         */
        __global__ void sort(Tree *tree, integer n, integer m);

        /**
         * @brief Kernel to get all particle's keys
         *
         * > Corresponding wrapper function: ::TreeNS::Kernel::Launch::getParticleKeys()
         *
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[out] keys input particle's keys
         * @param[in] maxLevel Tree maximum level
         * @param[in] n
         * @param[in] curveType SFC curve type (Lebesgue/Hilbert)
         */
        __global__ void getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n,
                                        Curve::Type curveType = Curve::lebesgue);

        /**
         * @brief Compute center of mass for all particles
         *
         * > Corresponding wrapper function: ::TreeNS::Kernel::Launch::globalCOM()
         *
         * @param[in] tree Tree class instance
         * @param[in] particles Particle class instance
         * @param[out] com Center of mass
         */
        __global__ void globalCOM(Tree *tree, Particles *particles, real com[DIM]);

        namespace Launch {

            /**
             * @brief Wrapper for ::TreeNS::Kernel::sumParticles()
             *
             * @return Wall time of execution
             */
            real sumParticles(Tree *tree);

            /**
             * @brief Wrapper for ::TreeNS::Kernel::getParticleKeys()
             *
             * @return Wall time of execution
             */
            real getParticleKeys(Tree *tree, Particles *particles, keyType *keys, integer maxLevel, integer n,
                                 Curve::Type curveType = Curve::lebesgue, bool time=false);

            /**
             * @brief Wrapper for ::TreeNS::Kernel::buildTree()
             *
             * @return Wall time of execution
             */
            real buildTree(Tree *tree, Particles *particles, integer n, integer m, bool time=false);

            real prepareSorting(Tree *tree, Particles *particles, integer n, integer m);

            real calculateCentersOfMass(Tree *tree, Particles *particles, integer n, integer level, bool time=false);

            /**
             * @brief Wrapper for ::TreeNS::Kernel::computeBoundingBox()
             *
             * @return Wall time of execution
             */
            real computeBoundingBox(Tree *tree, Particles *particles, integer *mutex, integer n,
                                          integer blockSize, bool time=false);

            /**
             * @brief Wrapper for ::TreeNS::Kernel::centerOfMass()
             *
             * @return Wall time of execution
             */
            real centerOfMass(Tree *tree, Particles *particles, integer n, bool time=false);

            /**
             * @brief Wrapper for ::TreeNS::Kernel::sort()
             *
             * @return Wall time of execution
             */
            real sort(Tree *tree, integer n, integer m, bool time=false);

            /**
             * @brief @brief Wrapper for ::TreeNS::Kernel::globalCOM()
             *
             * @return Wall time of execution
             */
            real globalCOM(Tree *tree, Particles *particles, real com[DIM]);

        }
    }
}

#if UNIT_TESTING
namespace UnitTesting {
    namespace Kernel {

        __global__ void test_localTree(Tree *tree, Particles *particles, int n, int m);

        namespace Launch {
            real test_localTree(Tree *tree, Particles *particles, int n, int m);
        }
    }
}
#endif
#endif // TARGET_GPU

#endif //MILUPHPC_TREE_CUH
