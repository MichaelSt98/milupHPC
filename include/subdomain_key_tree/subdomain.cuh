/**
 * @file subdomain.cu
 *
 * @brief Classes and Kernels handling subdomains that distribute the
 * workload among the MPI processes.
 *
 * @author Michael Staneker
 *
 */
#ifndef MILUPHPC_DOMAIN_CUH
#define MILUPHPC_DOMAIN_CUH

#include "../parameter.h"
#include "../cuda_utils/cuda_runtime.h"
#include "../cuda_utils/cuda_utilities.cuh"
#include "tree.cuh"

//TODO: necessary/reasonable to have Key class?
//class Key {
    //keyType *keys;
    //integer *maxLevel;
    //CUDA_CALLABLE_MEMBER Key();
    //CUDA_CALLABLE_MEMBER Key(keyType *keys, integer *maxLevel);
    //CUDA_CALLABLE_MEMBER ~Key();
    //CUDA_CALLABLE_MEMBER void set(keyType *keys, integer *maxLevel);*/
//};

// Forward declaration of DomainList class
class DomainList;
// Forward declaration of SubDomainKeyTree class
class SubDomainKeyTree;

namespace KeyNS {

    CUDA_CALLABLE_MEMBER void key2Char(keyType key, integer maxLevel, char *keyAsChar);
    CUDA_CALLABLE_MEMBER integer key2proc(keyType key, SubDomainKeyTree *subDomainKeyTree/*, Curve::Type curveType=Curve::lebesgue*/);
}

/**
 * SubDomainKeyTree class handling rank, number of processes and ranges
 */
class SubDomainKeyTree {

public:
    /// MPI rank
    integer rank;
    /// MPI number of processes
    integer numProcesses;
    /// Space-filling curve ranges, mapping key ranges/borders to MPI processes
    keyType *range;
    //keyType *hilbertRange;
    /// particle counter in dependence of MPI process(es)
    integer *procParticleCounter;

    /**
     * Default Constructor
     */
    CUDA_CALLABLE_MEMBER SubDomainKeyTree();
    /**
     * Constructor
     *
     * @param rank MPI rank
     * @param numProcesses MPI number of processes
     * @param range key ranges/borders
     * @param procParticleCounter // particle counter in dependence of MPI processes
     */
    CUDA_CALLABLE_MEMBER SubDomainKeyTree(integer rank, integer numProcesses, keyType *range,
                                          integer *procParticleCounter);
    /**
     * Destructor
     */
    CUDA_CALLABLE_MEMBER ~SubDomainKeyTree();
    /**
     * Setter
     *
     * @param rank MPI rank
     * @param numProcesses MPI number of processes
     * @param range key ranges/borders
     * @param procParticleCounter // particle counter in dependence of MPI processes
     */
    CUDA_CALLABLE_MEMBER void set(integer rank, integer numProcesses, keyType *range, integer *procParticleCounter);
    /**
     * Compute particle's MPI process belonging by it's key
     *
     * @param key input key, representing a particle/pseudo-particle/node
     * @return affiliated MPI process
     */
    CUDA_CALLABLE_MEMBER integer key2proc(keyType key/*, Curve::Type curveType=Curve::lebesgue*/);
    /**
     * Check whether key, thus particle, represents a domain list node
     *
     * @param key input key, representing a particle/pseudo-particle/node
     * @param maxLevel max tree level
     * @param level level of input key/particle
     * @param curveType space-filling curve type (Lebesgue/Hilbert)
     * @return whether domain list node or not
     */
    CUDA_CALLABLE_MEMBER bool isDomainListNode(keyType key, integer maxLevel, integer level,
                                               Curve::Type curveType=Curve::lebesgue);

};

namespace SubDomainKeyTreeNS {

    namespace Kernel {

        /**
         * Kernel call to setter
         *
         * @param subDomainKeyTree
         * @param rank
         * @param numProcesses
         * @param range
         * @param procParticleCounter
         */
        __global__ void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range,
                            integer *procParticleCounter);

        /**
         * Test kernel call (for debugging/testing purposes)
         *
         * @param subDomainKeyTree
         */
        __global__ void test(SubDomainKeyTree *subDomainKeyTree);

        /**
         * Kernel to build the domain tree
         *
         * Using the already built tree, marking domain list nodes
         * and adding them if necessary
         *
         * @param tree target tree instance
         * @param particles particle information
         * @param domainList domainList instance
         * @param n number of
         * @param m number of
         */
        __global__ void buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m);

        __global__ void buildDomainTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m, integer level);

        /**
         * Kernel to get all particle keys (and additional information for debugging purposes)
         *
         * @param subDomainKeyTree
         * @param tree
         * @param particles
         * @param keys
         * @param maxLevel
         * @param n
         * @param curveType
         */
        __global__ void getParticleKeys(SubDomainKeyTree *subDomainKeyTree, Tree *tree,
                                                        Particles *particles, keyType *keys, integer maxLevel,
                                                        integer n, Curve::Type curveType = Curve::lebesgue);

        /**
         * Kernel to check particle's belonging and count in dependence of belonging/process
         *
         * @param subDomainKeyTree
         * @param tree
         * @param particles
         * @param n
         * @param m
         * @param curveType
         */
        __global__ void particlesPerProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            integer n, integer m, Curve::Type curveType=Curve::lebesgue);

        /**
         * Kernel to mark particle's belonging
         *
         * @param subDomainKeyTree
         * @param tree
         * @param particles
         * @param n
         * @param m
         * @param sortArray
         * @param curveType
         */
        __global__ void markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                             integer n, integer m, integer *sortArray,
                                             Curve::Type curveType=Curve::lebesgue);

        namespace Launch {

            /**
             * Wrapped kernel call to setter
             *
             * @param subDomainKeyTree
             * @param rank
             * @param numProcesses
             * @param range
             * @param procParticleCounter
             */
            void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range,
                     integer *procParticleCounter);

            /**
             * Wrapped test kernel call (for debugging/testing purposes)
             *
             * @param subDomainKeyTree
             */
            void test(SubDomainKeyTree *subDomainKeyTree);

            /**
             * Wrapped kernel to build the domain tree
             *
             * @param tree
             * @param particles
             * @param domainList
             * @param n
             * @param m
             * @return
             */
            real buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m);

            real buildDomainTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m, integer level);

            /**
             * Wrapped kernel to get all particle keys (and additional information for debugging purposes)
             *
             * @param subDomainKeyTree
             * @param tree
             * @param particles
             * @param keys
             * @param maxLevel
             * @param n
             * @param curveType
             * @return
             */
            real getParticleKeys(SubDomainKeyTree *subDomainKeyTree, Tree *tree,
                                 Particles *particles, keyType *keys, integer maxLevel,
                                 integer n, Curve::Type curveType = Curve::lebesgue);

            /**
             * Wrapped Kernel to check particle's belonging and count in dependence of belonging/process
             *
             * @param subDomainKeyTree
             * @param tree
             * @param particles
             * @param n
             * @param m
             * @param curveType
             * @return
             */
            real particlesPerProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                integer n, integer m, Curve::Type curveType=Curve::lebesgue);

            /**
             * Wrapped kernel to mark particle's belonging
             *
             * @param subDomainKeyTree
             * @param tree
             * @param particles
             * @param n
             * @param m
             * @param sortArray
             * @param curveType
             * @return
             */
            real markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                 integer n, integer m, integer *sortArray,
                                                 Curve::Type curveType=Curve::lebesgue);
        }

    }

}

/**
 * Class to represent domain list nodes (lowest domain list nodes)
 */
class DomainList {

public:

    /// domain list node indices
    integer *domainListIndices;
    /// domain list node levels
    integer *domainListLevels;
    /// domain list node index, thus amount of domain list nodes
    integer *domainListIndex;
    /// domain list node counter, usable as buffer
    integer *domainListCounter;
    /// domain list node keys
    keyType *domainListKeys;
    /// sorted domain list node keys, usable as output for sorting the keys
    keyType *sortedDomainListKeys;
    /// concentrate domain list nodes, usable to reduce domain list indices in respect to some criterion
    integer *relevantDomainListIndices;
    ///
    integer *relevantDomainListLevels;
    ///
    integer *relevantDomainListProcess;

    /**
     * Constructor
     */
    CUDA_CALLABLE_MEMBER DomainList();
    /**
     * Constructor, passing pointer to member variables
     *
     * @param domainListIndices
     * @param domainListLevels
     * @param domainListIndex
     * @param domainListCounter
     * @param domainListKeys
     * @param sortedDomainListKeys
     * @param relevantDomainListIndices
     */
    CUDA_CALLABLE_MEMBER DomainList(integer *domainListIndices, integer *domainListLevels, integer *domainListIndex,
                                    integer *domainListCounter, keyType *domainListKeys, keyType *sortedDomainListKeys,
                                    integer *relevantDomainListIndices, integer *relevantDomainListLevels,
                                    integer *relevantDomainListProcess);
    /**
     * Setter, passing pointer to member variables
     *
     * @param domainListIndices
     * @param domainListLevels
     * @param domainListIndex
     * @param domainListCounter
     * @param domainListKeys
     * @param sortedDomainListKeys
     * @param relevantDomainListIndices
     */
    CUDA_CALLABLE_MEMBER void set(integer *domainListIndices, integer *domainListLevels, integer *domainListIndex,
                                  integer *domainListCounter, keyType *domainListKeys, keyType *sortedDomainListKeys,
                                  integer *relevantDomainListIndices, integer *relevantDomainListLevels,
                                  integer *relevantDomainListProcess);
    /**
     * Destructor
     */
    CUDA_CALLABLE_MEMBER ~DomainList();
};

namespace DomainListNS {

    namespace Kernel {
        /**
         * Kernel call to setter
         *
         * @param domainList
         * @param domainListIndices
         * @param domainListLevels
         * @param domainListIndex
         * @param domainListCounter
         * @param domainListKeys
         * @param sortedDomainListKeys
         * @param relevantDomainListIndices
         */
        __global__ void set(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                            integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                            keyType *sortedDomainListKeys, integer *relevantDomainListIndices,
                            integer *relevantDomainListLevels, integer *relevantDomainListProcess);

        /**
         * Info kernel (for debugging purposes)
         *
         * @param particles
         * @param domainList
         */
        __global__ void info(Particles *particles, DomainList *domainList);

        /**
         * Info kernel (for debugging purposes)
         *
         * @param particles
         * @param domainList
         * @param lowestDomainList
         */
        __global__ void info(Particles *particles, DomainList *domainList, DomainList *lowestDomainList);

        /**
         * Kernel to create the domain list
         *
         * @param subDomainKeyTree
         * @param domainList
         * @param maxLevel
         * @param curveType
         */
        __global__ void createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                         integer maxLevel, Curve::Type curveType = Curve::lebesgue);

        /**
         * Kernel to create the lowest domain list
         * 
         * @param subDomainKeyTree
         * @param tree
         * @param domainList
         * @param lowestDomainList
         * @param n
         * @param m
         */
        __global__ void lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                         DomainList *domainList, DomainList *lowestDomainList, integer n, integer m);

        namespace Launch {
            /**
             * Wrapped kernel call to setter
             *
             * @param domainList
             * @param domainListIndices
             * @param domainListLevels
             * @param domainListIndex
             * @param domainListCounter
             * @param domainListKeys
             * @param sortedDomainListKeys
             * @param relevantDomainListIndices
             */
            void set(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                     integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                     keyType *sortedDomainListKeys, integer *relevantDomainListIndices,
                     integer *relevantDomainListLevels, integer *relevantDomainListProcess);

            /**
             * Wrapped info kernel (for debugging purposes)
             *
             * @param particles
             * @param domainList
             * @return
             */
            real info(Particles *particles, DomainList *domainList);

            /**
             * Wrapped info kernel (for debugging purposes)
             *
             * @param particles
             * @param domainList
             * @param lowestDomainList
             * @return
             */
            real info(Particles *particles, DomainList *domainList, DomainList *lowestDomainList);

            /**
             * Wrapped kernel to create the domain list
             *
             * @param subDomainKeyTree
             * @param domainList
             * @param maxLevel
             * @param curveType
             * @return
             */
            real createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                  integer maxLevel, Curve::Type curveType = Curve::lebesgue);

            /**
             * Wrapped kernel to create the lowest domain list
             *
             * @param subDomainKeyTree
             * @param tree
             * @param domainList
             * @param lowestDomainList
             * @param n
             * @param m
             * @return
             */
            real lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                  DomainList *domainList, DomainList *lowestDomainList, integer n, integer m);
        }
    }

}

namespace ParticlesNS {

    __device__ bool applySphericalCriterion(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            real d, int index);

    __device__ bool applyCubicCriterion(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                        real d, int index);

    namespace Kernel {
        __global__ void mark2remove(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                    int *particles2remove, int *counter, int criterion, real d,
                                    int numParticles);

        namespace Launch {
            real mark2remove(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                             int *particles2remove, int *counter, int criterion, real d,
                             int numParticles);
        }
    }
}

namespace CudaUtils {
    namespace Kernel {
        template<typename T, typename U>
        __global__ void markDuplicatesTemp(Tree *tree, DomainList *domainList, T *array, U *entry1, U *entry2, U *entry3, integer *duplicateCounter, integer *child, int length);

        template <typename T, unsigned int blockSize>
        __global__ void reduceBlockwise(T *array, T *outputData, int n);

        template <typename T, unsigned int blockSize>
        __global__ void blockReduction(const T *indata, T *outdata);

        namespace Launch {
            template<typename T, typename U>
            real markDuplicatesTemp(Tree *tree, DomainList *domainList, T *array, U *entry1, U *entry2, U *entry3, integer *duplicateCounter, integer *child, int length);

            template <typename T, unsigned int blockSize>
            real reduceBlockwise(T *array, T *outputData, int n);

            template <typename T, unsigned int blockSize>
            real blockReduction(const T *indata, T *outdata);
        }
    }
}

namespace Physics {
    namespace Kernel {

        template <unsigned int blockSize>
        __global__ void calculateAngularMomentumBlockwise(Particles *particles, real *outputData, int n);

        template <unsigned int blockSize>
        __global__ void sumAngularMomentum(const real *indata, real *outdata);

        __global__ void kineticEnergy(Particles *particles, int n);

        namespace Launch {
            template <unsigned int blockSize>
            real calculateAngularMomentumBlockwise(Particles *particles, real *outputData, int n);

            template <unsigned int blockSize>
            real sumAngularMomentum(const real *indata, real *outdata);

            real kineticEnergy(Particles *particles, int n);
        }
    }
}

#endif //MILUPHPC_DOMAIN_CUH
