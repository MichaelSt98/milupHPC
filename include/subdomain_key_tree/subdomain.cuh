/**
 * @file subdomain.cuh
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
#include "../helper_handler.h"
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

/// Key (keyType) related functions and kernels
namespace KeyNS {

    /**
     * @brief Convert a key to a char for printing.
     *
     * @param[in] key Key to be converted
     * @param[in] maxLevel maximum (tree) level
     * @param[out] keyAsChar key as char
     */
    CUDA_CALLABLE_MEMBER void key2Char(keyType key, integer maxLevel, char *keyAsChar);

    /**
     * @brief Convert the key to the corresponding process
     *
     * The mapping from a key to a MPI process via ranges can be implemented by checking in between
     * which successive entries of the range the key is located.
     *
     * @param key Key to be evaluated
     * @param subDomainKeyTree SubDomainKeyTree class instance
     * @return Process key belongs to
     */
    CUDA_CALLABLE_MEMBER integer key2proc(keyType key, SubDomainKeyTree *subDomainKeyTree/*, Curve::Type curveType=Curve::lebesgue*/);
}

/**
 * @brief SubDomainKeyTree class handling rank, number of processes and ranges
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
     * @brief Default Constructor.
     */
    CUDA_CALLABLE_MEMBER SubDomainKeyTree();
    /**
     * @brief Constructor.
     *
     * @param rank MPI rank
     * @param numProcesses MPI number of processes
     * @param range key ranges/borders
     * @param procParticleCounter // particle counter in dependence of MPI processes
     */
    CUDA_CALLABLE_MEMBER SubDomainKeyTree(integer rank, integer numProcesses, keyType *range,
                                          integer *procParticleCounter);
    /**
     * @brief Destructor.
     */
    CUDA_CALLABLE_MEMBER ~SubDomainKeyTree();

    /**
     * @brief Setter.
     *
     * @param rank MPI rank
     * @param numProcesses MPI number of processes
     * @param range key ranges/borders
     * @param procParticleCounter // particle counter in dependence of MPI processes
     */
    CUDA_CALLABLE_MEMBER void set(integer rank, integer numProcesses, keyType *range, integer *procParticleCounter);

    /**
     * @brief Compute particle's MPI process belonging by it's key.
     *
     * @param key input key, representing a particle/pseudo-particle/node
     * @return affiliated MPI process
     */
    CUDA_CALLABLE_MEMBER integer key2proc(keyType key/*, Curve::Type curveType=Curve::lebesgue*/);

    /**
     * @brief Check whether key, thus particle, represents a domain list node.
     *
     * @param key input key, representing a particle/pseudo-particle/node
     * @param maxLevel max tree level
     * @param level level of input key/particle
     * @param curveType space-filling curve type (Lebesgue/Hilbert)
     * @return whether domain list node or not
     */
    CUDA_CALLABLE_MEMBER bool isDomainListNode(keyType key, integer maxLevel, integer level,
                                               Curve::Type curveType=Curve::lebesgue);

    void buildDomainTree(Tree *tree, DomainList *domainList);

    void buildDomainTree(Tree *tree, DomainList *domainList, int level, keyType key);

};

#if TARGET_GPU
/// SubDomainKeyTree related functions and kernels
namespace SubDomainKeyTreeNS {

    /// Kernels
    namespace Kernel {

        /**
         * @brief Kernel call to setter.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::set()
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance
         * @param rank MPI rank
         * @param numProcesses MPI number of processes
         * @param range SFC curve range to MPI process mapping
         * @param procParticleCounter Number of particles per MPI process
         */
        __global__ void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range,
                            integer *procParticleCounter);

        /**
         * @brief Test kernel call (for debugging/testing purposes).
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::test()
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance
         */
        __global__ void test(SubDomainKeyTree *subDomainKeyTree);

        /**
         * @brief Kernel to build the domain tree.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::buildDomainTree()
         *
         * Using the already built tree, marking domain list nodes
         * and adding them if necessary.
         *
         * The common coarse tree construction takes care of the existence of the domain list nodes and saves
         * this information in an instance of the DomainList class by traversing the tree
         * specified by the keys derived in ::TreeNS::Kernel::createDomainList() and distinguishes three cases:
         *
         * * The corresponding node exists, thus nothing to do
         * * The corresponding node exists, but is a leaf, which is not allowed, thus insert pseudo-particle in between
         * * The corresponding node does not exist, thus create/request
         *
         * Subsequent the index of the found domain list node is saved. This is possible since the key contains the
         * path of the node in the tree, which can be used for traversing.
         *
         * @param tree Target Tree class instance
         * @param particles Particle class instance/information
         * @param domainList domainList instance
         * @param n number of
         * @param m number of
         */
        __global__ void buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m);

        __global__ void buildDomainTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m, integer level);

        /**
         * @brief Kernel to get all particle keys (and additional information for debugging purposes).
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::getParticleKeys()
         *
         * @param subDomainKeyTree SubdomainKeyTree class instance
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param keys Particle keys
         * @param maxLevel Tree maximum level
         * @param n number of particles
         * @param curveType SFC curve type (Lebesgue/Hilbert)
         */
        __global__ void getParticleKeys(SubDomainKeyTree *subDomainKeyTree, Tree *tree,
                                                        Particles *particles, keyType *keys, integer maxLevel,
                                                        integer n, Curve::Type curveType = Curve::lebesgue);

        /**
         * @brief Kernel to check particle's belonging and count in dependence of belonging/process.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::particlesPerProcess()
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param n
         * @param m
         * @param curveType SFC curve type (Lebesgue/Hilbert)
         */
        __global__ void particlesPerProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            integer n, integer m, Curve::Type curveType=Curve::lebesgue);

        /**
         * @brief Kernel to mark particle's belonging.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::markParticlesProcess()
         *
         * To assign particles to the correct process, the key for each particle is computed and translated to a
         * process via ::KeyNS::key2proc(). This information is saved in an array which can be used for
         * sorting enabling the subsequent copying into contiguous memory and exchanging via MPI.
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param n
         * @param m
         * @param sortArray
         * @param curveType SFC curve type (Lebesgue/Hilbert)
         */
        __global__ void markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                             integer n, integer m, integer *sortArray,
                                             Curve::Type curveType=Curve::lebesgue);

        /**
         * @brief Zero domain list nodes.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::zeroDomainListNodes()
         *
         * @param particles Particles class instance
         * @param domainList DomainList class instance/information about domain list nodes
         * @param lowestDomainList DomainList class instance/information about lowest domain list nodes
         */
        __global__ void zeroDomainListNodes(Particles *particles, DomainList *domainList,
                                            DomainList *lowestDomainList);

        /**
         * @brief Prepare lowest domain exchange via MPI by copying to contiguous memory.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::prepareLowestDomainExchange()
         *
         * @param particles Particles class instance
         * @param lowestDomainList DomainList class instance/information about lowest domain list nodes
         * @param helper
         * @param entry
         */
         template <typename T>
        __global__ void prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                                    T *buffer, Entry::Name entry);

        /**
         * @brief Update lowest domain list nodes.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::updateLowestDomainListNodes()
         *
         * @param particles Particles class instance
         * @param lowestDomainList DomainList class instance/information about lowest domain list nodes
         * @param helper
         * @param domainListSize
         * @param entry
         */
        template <typename T>
        __global__ void updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                                    T *buffer, Entry::Name entry);

        /**
         * @brief Compute/Find lowest domain list nodes.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::compLowestDomainListNodes()
         *
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param lowestDomainList DomainList class instance/information about lowest domain list nodes
         */
        __global__ void compLowestDomainListNodes(Tree *tree, Particles *particles, DomainList *lowestDomainList);

        /**
         * @brief Compute local (tree) pseudo particles.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::compLocalPseudoParticles()
         *
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param domainList DomainList class instance/information about domain list nodes
         * @param n
         */
        __global__ void compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n);

        /**
         * @brief Compute domain list pseudo particles (per level).
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::compDomainListPseudoParticlesPerLevel()
         *
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param domainList DomainList class instance/information about domain list nodes
         * @param lowestDomainList DomainList class instance/information about lowest domain list nodes
         * @param n
         * @param level Tree level
         */
        __global__ void compDomainListPseudoParticlesPerLevel(Tree *tree, Particles *particles, DomainList *domainList,
                                                              DomainList *lowestDomainList, int n, int level);

        __global__ void compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                                      DomainList *lowestDomainList, int n);

        /**
         * @brief Repair tree by removing received and inserted (pseudo-)particles.
         *
         * > Corresponding wrapper function: ::SubDomainKeyTreeNS::Kernel::Launch::repairTree()
         *
         * Deleting the received (pseudo-)particles is inevitable for simulations using a predictor-corrector
         * scheme or for simulations with self-gravity and SPH. Key part of this is the knowledge of whether a
         * (pseudo-)particle is received or intrinsically belongs to the process by its index. This information
         * is saved in the variables `toDeleteLeaf` and `toDeleteNode`.
         * Only the domain list nodes need to be handled separately, since children of these nodes if they do
         * not belong to the corresponding process need to be deleted as well.
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param domainList DomainList class instance/information about domain list nodes
         * @param lowestDomainList DomainList class instance/information about lowest domain list nodes
         * @param n
         * @param m
         * @param curveType SFC curve type (Lebesgue/Hilbert)
         */
        __global__ void repairTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                   DomainList *domainList, DomainList *lowestDomainList,
                                   int n, int m, Curve::Type curveType);

        // notes:
        // - using Helper::keyTypeBuffer as keyHistRanges
        __global__ void createKeyHistRanges(Helper *helper, integer bins);

        // notes:
        // - using Helper::keyTypeBuffer as keyHistRanges
        // - using Helper::integerBuffer as keyHistCounts
        __global__ void keyHistCounter(Tree *tree, Particles *particles, SubDomainKeyTree *subDomainKeyTree,
                                       Helper *helper,
                /*keyType *keyHistRanges, integer *keyHistCounts,*/ int bins, int n,
                                       Curve::Type curveType=Curve::lebesgue);

        // notes:
        // - using Helper::keyTypeBuffer as keyHistRanges
        // - using Helper::integerBuffer as keyHistCounts
        __global__ void calculateNewRange(SubDomainKeyTree *subDomainKeyTree, Helper *helper,
                /*keyType *keyHistRanges, integer *keyHistCounts,*/ int bins, int n,
                                          Curve::Type curveType=Curve::lebesgue);


        /// Wrapped kernels
        namespace Launch {

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::set().
             */
            void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range,
                     integer *procParticleCounter);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::test().
             */
            void test(SubDomainKeyTree *subDomainKeyTree);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::buildDomainTree().
             *
             * @return Wall time of execution
             */
            real buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m);

            real buildDomainTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m, integer level);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::getParticleKeys().
             *
             * @return Wall time of execution
             */
            real getParticleKeys(SubDomainKeyTree *subDomainKeyTree, Tree *tree,
                                 Particles *particles, keyType *keys, integer maxLevel,
                                 integer n, Curve::Type curveType = Curve::lebesgue);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::particlesPerProcess().
             *
             * @return Wall time of execution
             */
            real particlesPerProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                integer n, integer m, Curve::Type curveType=Curve::lebesgue);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::markParticlesPerProcess().
             *
             * @return Wall time of execution
             */
            real markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                 integer n, integer m, integer *sortArray,
                                                 Curve::Type curveType=Curve::lebesgue);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::zeroDomainListNodes().
             *
             * @return Wall time of execution
             */
            real zeroDomainListNodes(Particles *particles, DomainList *domainList, DomainList *lowestDomainList);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::prepareLowestDomainExchange().
             *
             * @return Wall time of execution
             */
            template <typename T>
            real prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                             T *buffer, Entry::Name entry);


            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::updateLowestDomainListNodes().
             *
             * @return Wall time of execution
             */
            template <typename T>
            real updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                             T *buffer, Entry::Name entry);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::compLowestDomainListNodes().
             *
             * @return Wall time of execution
             */
            real compLowestDomainListNodes(Tree *tree, Particles *particles, DomainList *lowestDomainList);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::compLocalPseudoParticles().
             *
             * @return Wall time of execution
             */
            real compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::compDomainListPseudoParticlesPerLevel().
             *
             * @return Wall time of execution
             */
            real compDomainListPseudoParticlesPerLevel(Tree *tree, Particles *particles, DomainList *domainList,
                                                       DomainList *lowestDomainList, int n, int level);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::compDomainListPseudoParticles().
             *
             * @return Wall time of execution
             */
            real compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                               DomainList *lowestDomainList, int n);

            /**
             * @brief Wrapper for ::SubDomainKeyTreeNS::Kernel::repairTree().
             *
             * @return Wall time of execution
             */
            real repairTree(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                            DomainList *domainList, DomainList *lowestDomainList,
                            int n, int m, Curve::Type curveType);

            real createKeyHistRanges(Helper *helper, integer bins);

            real keyHistCounter(Tree *tree, Particles *particles, SubDomainKeyTree *subDomainKeyTree,
                                Helper *helper, int bins, int n, Curve::Type curveType=Curve::lebesgue);

            real calculateNewRange(SubDomainKeyTree *subDomainKeyTree, Helper *helper, int bins, int n,
                                   Curve::Type curveType=Curve::lebesgue);
        }

    }

}

#endif

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

    integer *relevantDomainListOriginalIndex;

    real *borders;

    /**
     * @brief Constructor.
     */
    CUDA_CALLABLE_MEMBER DomainList();

    /**
     * @brief Constructor, passing pointer to member variables.
     *
     * @param domainListIndices Indices of the domain list nodes in Particles class instance
     * @param domainListLevels Levels of the domain list nodes within (to be built) Tree
     * @param domainListIndex
     * @param domainListCounter
     * @param domainListKeys Keys of the domain list nodes
     * @param sortedDomainListKeys Sorted (or buffer for sorting) of the domain list nodes keys
     * @param relevantDomainListIndices
     */
    CUDA_CALLABLE_MEMBER DomainList(integer *domainListIndices, integer *domainListLevels, integer *domainListIndex,
                                    integer *domainListCounter, keyType *domainListKeys, keyType *sortedDomainListKeys,
                                    integer *relevantDomainListIndices, integer *relevantDomainListLevels,
                                    integer *relevantDomainListProcess);
    /**
     * @brief Setter, passing pointer to member variables.
     *
     * @param domainListIndices Indices of the domain list nodes in Particles class instance
     * @param domainListLevels Levels of the domain list nodes within (to be built) Tree
     * @param domainListIndex
     * @param domainListCounter
     * @param domainListKeys Keys of the domain list nodes
     * @param sortedDomainListKeys Sorted (or buffer for sorting) of the domain list nodes keys
     * @param relevantDomainListIndices
     */
    CUDA_CALLABLE_MEMBER void set(integer *domainListIndices, integer *domainListLevels, integer *domainListIndex,
                                  integer *domainListCounter, keyType *domainListKeys, keyType *sortedDomainListKeys,
                                  integer *relevantDomainListIndices, integer *relevantDomainListLevels,
                                  integer *relevantDomainListProcess);

    CUDA_CALLABLE_MEMBER void setBorders(real *borders, integer *relevantDomainListOriginalIndex);

    /**
     * @brief Destructor.
     */
    CUDA_CALLABLE_MEMBER ~DomainList();
};

#if TARGET_GPU
namespace DomainListNS {

    namespace Kernel {

        /**
         * @brief Kernel call to setter.
         *
         * > Corresponding wrapper function: ::DomainListNS::Kernel::Launch::set()
         *
         * @param domainList DomainList class instance (to be constructed)
         * @param domainListIndices Indices of the domain list nodes in Particles class instance
         * @param domainListLevels Levels of the domain list nodes within (to be built) Tree
         * @param domainListIndex
         * @param domainListCounter
         * @param domainListKeys Keys of the domain list nodes in Particles class instance
         * @param sortedDomainListKeys
         * @param relevantDomainListIndices
         */
        __global__ void set(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                            integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                            keyType *sortedDomainListKeys, integer *relevantDomainListIndices,
                            integer *relevantDomainListLevels, integer *relevantDomainListProcess);

        __global__ void setBorders(DomainList *domainList, real *borders, integer *relevantDomainListOriginalIndex);

        /**
         * @brief Info kernel (for debugging purposes).
         *
         * > Corresponding wrapper function: ::DomainListNS::Kernel::Launch::info()
         *
         * @param particles
         * @param domainList
         */
        __global__ void info(Particles *particles, DomainList *domainList);

        /**
         * @brief Info kernel (for debugging purposes).
         *
         * > Corresponding wrapper function: ::DomainListNS::Kernel::Launch::info()
         *
         * @param particles
         * @param domainList
         * @param lowestDomainList
         */
        __global__ void info(Particles *particles, DomainList *domainList, DomainList *lowestDomainList);

        /**
         * @brief Kernel to create the domain list.
         *
         * > Corresponding wrapper function: ::DomainListNS::Kernel::Launch::createDomainList()
         *
         * In order to create or derive the domain list nodes from the current ranges a (non-existent) tree
         * is traversed by traversing the SFC via keys, whereas irrelevant parts of the keys are skipped.
         * If a certain node is a domain list node, the key as well as the level is saved in order to assign
         * this domain list node to a real node later on.
         *
         * @param subDomainKeyTree
         * @param domainList
         * @param maxLevel
         * @param curveType
         */
        __global__ void createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                         integer maxLevel, Curve::Type curveType = Curve::lebesgue);

        /**
         * @brief Kernel to create the lowest domain list.
         *
         * > Corresponding wrapper function: ::DomainListNS::Kernel::Launch::lowestDomainList()
         *
         * Lowest domain list nodes are identified by checking the common coarse tree node's children.
         * If at least one child itself is a part of the common coarse tree the corresponding node is not a
         * lowest domain list node.
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
             * @brief Wrapper for ::DomainListNS::Kernel::set().
             */
            void set(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                     integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                     keyType *sortedDomainListKeys, integer *relevantDomainListIndices,
                     integer *relevantDomainListLevels, integer *relevantDomainListProcess);

            void setBorders(DomainList *domainList, real *borders, integer *relevantDomainListOriginalIndex);

            /**
             * @brief Wrapper for ::DomainListNS::Kernel::info().
             *
             * @return Wall time of execution
             */
            real info(Particles *particles, DomainList *domainList);

            real info(Particles *particles, DomainList *domainList, DomainList *lowestDomainList);

            /**
             * @brief Wrapper for ::DomainListNS::Kernel::createDomainList().
             *
             * @return Wall time of execution
             */
            real createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                  integer maxLevel, Curve::Type curveType = Curve::lebesgue);

            /**
             * @brief Wrapper for ::DomainListNS::Kernel::lowestDoainList().
             *
             * @return Wall time of execution
             */
            real lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                  DomainList *domainList, DomainList *lowestDomainList, integer n, integer m);

        }
    }

}
#endif // TARGET_GPU


#if TARGET_GPU
/// Particle class related functions and kernels
namespace ParticlesNS {

    /**
     * @brief Check whether particle(s) are within sphere from simulation center.
     *
     * @param subDomainKeyTree SubDomainKeyTree class instance
     * @param tree Tree class instance
     * @param particles Particles class instance
     * @param d distance/diameter/radius
     * @param index Particle index within Particles class instance
     * @return Whether particle within sphere from simulation center
     */
    __device__ bool applySphericalCriterion(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            real d, int index);

    /**
     * @brief Check whether particle(s) are within cube from simulation center.
     *
     * @param subDomainKeyTree SubDomainKeyTree class instance
     * @param tree Tree class instance
     * @param particles Particles class instance
     * @param d distance/cube length
     * @param index Particle index within Particles class instance
     * @return Whether particle within cube from simulation center
     */
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

/// Physics related functions and kernels
namespace Physics {
    namespace Kernel {

        /**
         * @brief Calculate angular momentum for all particles (per block).
         *
         * > Corresponding wrapper function: ::Physics::Kernel::Launch::calculateAngularMomentumBlockwise()
         *
         * @tparam[in] blockSize
         * @param[in] particles
         * @param[out] outputData
         * @param[in] n
         */
        template <unsigned int blockSize>
        __global__ void calculateAngularMomentumBlockwise(Particles *particles, real *outputData, int n);

        /**
         * @brief Calculate angular momentum: sum over blocks.
         *
         * > Corresponding wrapper function: ::Physics::Kernel::Launch::sumAngularMomentum()
         *
         * @tparam[in] blockSize
         * @param[in] indata
         * @param[out] outdata
         */
        template <unsigned int blockSize>
        __global__ void sumAngularMomentum(const real *indata, real *outdata);

        /**
         * @brief Calculate kinetic energy.
         *
         * > Corresponding wrapper function: ::Physics::Kernel::Launch::kineticEnergy()
         *
         * @param particles
         * @param n
         */
        __global__ void kineticEnergy(Particles *particles, int n);

        namespace Launch {
            /**
             * @brief Wrapper for: ::Physics::Kernel::calculateAngularMomentumBlockwise().
             *
             * @return Wall time of execution
             */
            template <unsigned int blockSize>
            real calculateAngularMomentumBlockwise(Particles *particles, real *outputData, int n);

            /**
             * @brief Wrapper for: ::Physics::Kernel::sumAngularMomentum().
             *
             * @return Wall time of execution
             */
            template <unsigned int blockSize>
            real sumAngularMomentum(const real *indata, real *outdata);

            /**
             * @brief Wrapper for: ::Physics::Kernel::kineticEnergy().
             *
             * @return Wall time of execution
             */
            real kineticEnergy(Particles *particles, int n);
        }
    }
}

#endif // TARGET_GPU

#endif //MILUPHPC_DOMAIN_CUH
