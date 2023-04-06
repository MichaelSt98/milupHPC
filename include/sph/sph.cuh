/**
 * @file sph.cuh
 * @brief SPH related functionalities and kernels.
 *
 * Including:
 *
 * * fixed-radius near neighbor search (FRNN)
 * * finding particles to be sent
 * * inserting received particles
 * * ...
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_SPH_CUH
#define MILUPHPC_SPH_CUH

#include "../particles.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "../helper.cuh"
#include "../materials/material.cuh"
#include <float.h>

#include <boost/mpi.hpp>
#include <assert.h>

/// SPH related functions and kernels.
namespace SPH {

    /**
     * @deprecated
     */
    void exchangeParticleEntry(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend, integer *sendLengths,
                               integer *receiveLengths, integer numParticlesLocal);

    /// SPH related (CUDA) kernels.
    namespace Kernel {

        /**
         * @brief Create necessary ghost particles for periodic boundary conditions
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::createGhostsPeriodic
         *
         * This function is only used for SPH simulations and when PERIODIC_BOUNDARIES is true.
         * The container `ghostParticles` is filled with the required particles to enforce periodic boundaries.
         *
         * @param[in] tree Tree class instance needed for bounding box
         * @param[in] particles Particles class instance
         * @param [out] ghostParticleIndices Indices at which ghost particles are located
         * @param[out] ghostParticles IntegratedParticles instance holding ghost particles
         * @param[out] numGhosts Number of ghost particles
         * @param[in] searchRadius Radius which is the maximum distance to the borders for a particle
         *                         to be mirrored as ghost, typically smoothing length
         * @param[in] numParticlesLocal number of particles on this process
         */
         __global__ void createGhostsPeriodic(Tree *tree, Particles *particles, integer *ghostParticleIndices,
                                              IntegratedParticles *ghostParticles, integer &numGhosts,
                                              real searchRadius, integer numParticlesLocal);

        /**
         * @brief Fixed-radius near neighbor search (brute-force method).
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::fixedRadiusNN_bruteForce()
         *
         * @warning This implementation is primarily for comparison purposes and not for production usage!
         *
         * Straight-forward brute-force method for the fixed radius near neighbor search!
         *
         * Alternative methods:
         *
         * * ::SPH::Kernel::fixedRadiusNN() as a tree based algorithm
         * * ::SPH::Kernel::fixedRadiusNN_withinBox() as a (more sophisticated) tree based algorithm
         *
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[out] interactions interaction partners
         * @param[in] numParticlesLocal number of local particles
         * @param[in] numParticles number of particles in total
         * @param[in] numNodes number of nodes
         */
        __global__ void fixedRadiusNN_bruteForce(Tree *tree, Particles *particles, integer *interactions,
                                                 integer numParticlesLocal, integer numParticles, integer numNodes);

        /**
         * @brief Fixed-radius near neighbor search (default method via explicit stack).
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::fixedRadiusNN()
         *
         * Alternative methods:
         *
         * * ::SPH::Kernel::fixedRadiusNN_bruteForce() as brute-force method
         * * ::SPH::Kernel::fixedRadiusNN_withinBox() as a (more sophisticated) tree based algorithm
         *
         * Besides the straightforward brute-force approach to find the neighbors within the smoothing length as
         * presented in ::SPH::Kernel::fixedRadiusNN_bruteForce(), there are two more sophisticated approaches
         * for the FRNN search via the tree implemented. This algorithm utilizes an explicit stack for each
         * particle to traverse the tree. In case of the node being a particle it is checked whether the distance
         * is smaller than the smoothing length, so that this particle is added to the interaction list. In the
         * other case of the node being a pseudo-particle, it is tested whether particles within the cell of
         * this pseudo-particle are possibly within the range of the smoothing length and consequently either
         * the node added to the stack or the traversal terminated for this node. This possibly early termination
         * of traversing entire sub-trees is the key component for possible performance advantages in comparison
         * to the brute-force approach.
         *
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[out] interactions interaction partners
         * @param radius
         * @param[in] numParticlesLocal number of local particles
         * @param[in] numParticles number of particles in total
         * @param[in] numNodes number of nodes
         */
        __global__ void fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, real radius,
                                      integer numParticlesLocal, integer numParticles, integer numNodes);

        /**
         * @brief Fixed-radius near neighbor search (nested stack method).
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::fixedRadiusNN_withinBox()
         *
         * Alternative methods:
         *
         * * ::SPH::Kernel::fixedRadiusNN_bruteForce() as brute-force method
         * * ::SPH::Kernel::fixedRadiusNN() as a tree based algorithm
         *
         * This algorithm is similar to ::SPH::Kernel::fixedRadiusNN(), but in addition for checking whether
         * the cell of a pseudo-particle is possibly within the smoothing length of the particle for which the
         * neighbors are searched for, it is checked whether the cell or box of this pseudo-particle may be
         * in entirely within the range of the smoothing length. If this is fulfilled, all the particles beneath
         * this pseudo-particle are added to the interaction list. This is done by a second explicit stack for
         * which the primary explicit stack can be reused as shown
         *
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[out] interactions interaction partners
         * @param[in] numParticlesLocal number of local particles
         * @param[in] numParticles number of particles in total
         * @param[in] numNodes number of nodes
         */
        __global__ void fixedRadiusNN_withinBox(Tree *tree, Particles *particles, integer *interactions,
                                                integer numParticlesLocal, integer numParticles, integer numNodes);

        /**
         * @brief Fixed-radius near neighbor search (brute-force method).
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::fixedRadiusNN_sharedMemory()
         *
         * @warning Experimental for now!
         *
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[out] interactions interaction partners
         * @param[in] numParticlesLocal number of local particles
         * @param[in] numParticles number of particles in total
         * @param[in] numNodes number of nodes
         */
        __global__ void fixedRadiusNN_sharedMemory(Tree *tree, Particles *particles, integer *interactions,
                                                   integer numParticlesLocal, integer numParticles, integer numNodes);

        /**
         * @brief Fixed-radius near neighbor search for iteratively finding appropriate smoothing length.
         *
         * > Corresponding wrapper function: ::SPH::Kernel::Launch::fixedRadiusNN_variableSML()
         *
         * This function is not for finding the near neighbor for interacting, but for finding the correct or adequate
         * smoothing length in dependence of the desired number of interaction partners!
         *
         * @param[in] materials Material class instance
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[out] interactions interaction partners
         * @param[in] numParticlesLocal number of local particles
         * @param[in] numParticles number of particles in total
         * @param[in] numNodes number of nodes
         */
        __global__ void fixedRadiusNN_variableSML(Material *materials, Tree *tree, Particles *particles, integer *interactions,
                                                  integer numParticlesLocal, integer numParticles,
                                                  integer numNodes);

        /**
         * @brief Redo the neighbor search (FRNN).
         *
         * @todo test appropriately
         *
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[in] particleId particle identifier for the particle to redo the neighbor search
         * @param[out] interactions interaction partners
         * @param[in] radius smoothing length
         * @param[in] numParticles number of particles
         * @param[in] numNodes number of nodes
         */
        __device__ void redoNeighborSearch(Tree *tree, Particles *particles, int particleId,
                                             int *interactions, real radius, integer numParticles, integer numNodes);

        /**
         * @brief Find the relevant (lowest) domain list nodes as preparation for finding particles to be exchanged
         * between processes.
         *
         * This function identifies the (lowest) domain list nodes that do not belong to the corresponding process as
         * necessary subsequent measure to find particles that need to be exchanged between processes to grant
         * correctness of SPH forces.
         *
         * @param[in] subDomainKeyTree SubDomainKeyTree class instance
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[in, out] lowestDomainList DomainList class instance describing the lowest domain list nodes
         * @param[in] curveType Space-filling curve type used (see ::Curve)
         */
        __global__ void compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                  DomainList *lowestDomainList, Curve::Type curveType);

        /**
         * @brief Find the particles that need to be exchanged between processes to grant
         * correctness of SPH forces.
         *
         * Check for each particle and process that is not the corresponding process whether the particle might be
         * needed for the SPH method since another particle on a distinct process might be within the smoothing length.
         *
         * @param[in] subDomainKeyTree SubDomainKeyTree class instance
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[in] lowestDomainList DomainList class instance describing the lowest domain list nodes
         * @param[out] sendIndices Particles or rather their indices to be sent
         * @param[in] searchRadius Distance to different domain as condition for sending this particle
         * @param[in] n number of particles
         * @param[in] m number of nodes
         * @param[in] relevantIndex (Lowest) Domain list node index to be investigated/tested
         * @param[in] curveType Space-filling curve type used (see ::Curve)
         */
        __global__ void symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *lowestDomainList, integer *sendIndices, real searchRadius,
                                      integer n, integer m, integer relevantIndex,
                                      Curve::Type curveType);

        __global__ void symbolicForce_test(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                           DomainList *lowestDomainList, integer *sendIndices, real searchRadius,
                                           integer n, integer m, integer relevantProc, integer relevantIndicesCounter,
                                           integer relevantIndexOld, Curve::Type curveType);

        __global__ void symbolicForce_test2(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            DomainList *domainList, integer *sendIndices, real searchRadius,
                                            integer n, integer m, integer relevantProc, integer relevantIndicesCounter,
                                            Curve::Type curveType);

        /**
         * @brief Collect the found particles into contiguous memory in order to facilitate sending via MPI.
         *
         * Collect all the by ::SPH::Kernel::symbolicForce() previously found particles or rather indices of the
         * particles into contiguous memory by copying to a buffer array.
         *
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[in] sendIndices Not contiguous particles to be sent/Array with particle indices marked to be sent
         * @param[out] particles2Send Contiguous collection of indices to be sent
         * @param[out] particlesCount Amount of particles to be sent
         * @param[in] n Number of particles
         * @param[in] length
         * @param[in] curveType Space-filling curve type used (see ::Curve)
         */
        __global__ void collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                           integer *particles2Send, integer *particlesCount,
                                           integer n, integer length, Curve::Type curveType);

        __global__ void collectSendIndices_test2(Tree *tree, Particles *particles, integer *sendIndices,
                                                 integer *particles2Send, integer *particlesCount,
                                                 integer numParticlesLocal, integer numParticles,
                                                 integer treeIndex, int currentProc, Curve::Type curveType);

        /**
         * @deprecated
         */
        __global__ void particles2Send(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                       DomainList *domainList, DomainList *lowestDomainList, integer maxLevel,
                                       integer *toSend, integer *sendCount, integer *alreadyInserted,
                                       integer insertOffset,
                                       integer numParticlesLocal, integer numParticles, integer numNodes, real radius,
                                       Curve::Type curveType = Curve::lebesgue);

        /**
         * @deprecated
         */
        __global__ void collectSendIndicesBackup(integer *toSend, integer *toSendCollected, integer count);

        /**
         * @deprecated
         */
        __global__ void collectSendEntriesBackup(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend,
                                           integer *sendIndices, integer *sendCount, integer totalSendCount,
                                           integer insertOffset);

        /**
         * @brief Insert the received particles into the local tree.
         *
         * Insert the previously received particles into the local tree as similar approach
         * to the actual tree creation in ::TreeNS::Kernel::buildTree().
         *
         * @param[in] subDomainKeyTree SubdomainKeyTree class instance
         * @param[in, out] tree Tree class instance
         * @param[in, out] particles Particles class instance
         * @param[in] domainList DomainList class instance
         * @param[in] lowestDomainList DomainList class instance describing the lowest domain list nodes
         * @param[in] n Number of particles
         * @param[in] m Number of nodes
         */
        __global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m);


        /**
         * @deprecated
         */
        __global__ void calculateCentersOfMass(Tree *tree, Particles *particles, integer level);

        /**
         * @brief Determine the search radius needed for ::SPH::Kernel::symbolicForce().
         *
         * Determines the minimal distance to each domain/process that is not the domain/process of the particle itself.
         *
         * @param[in] subDomainKeyTree SubDomainKeyTree class instance
         * @param[in] tree Tree class instance
         * @param[in] particles Particles class instance
         * @param[in] domainList DomainList class instance
         * @param[in] lowestDomainList DomainList class instance describing the lowest domain list nodes
         * @param[out] searchRadii search radii/ search radius for each particle
         * @param[in] n Number of particles
         * @param[in] m Number of nodes
         * @param[in] curveType Space-filling curve type used (see ::Curve)
         */
        __global__ void determineSearchRadii(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                             DomainList *domainList, DomainList *lowestDomainList, real *searchRadii,
                                             int n, int m, Curve::Type curveType);

        /**
         * @brief Info/Debug kernel.
         *
         * @param tree Tree class instance
         * @param particles Particles class instance
         * @param helper Helper class instance
         * @param numParticlesLocal Number of local particles/Particles on this process
         * @param numParticles Number of particles (in total)
         * @param numNodes Number of nodes
         */
        __global__ void info(Tree *tree, Particles *particles, Helper *helper,
                             integer numParticlesLocal, integer numParticles, integer numNodes);

        /// SPH related (CUDA) kernel wrappers.
        namespace Launch {

            /**
             * @brief Wrapper for ::SPH::Kernel::createGhostsPeriodic
             *
             * @return Wall time for kernel execution
             */
             real createGhostsPeriodic(Tree *tree, Particles *particles, integer *ghostParticleIndices,
                                       IntegratedParticles *ghostParticles, integer &numGhosts,
                                       real searchRadius, integer numParticlesLocal);

            /**
             * @brief Wrapper for ::SPH::Kernel::fixedRadiusNN_bruteForce().
             *
             * @return Wall time for kernel execution
             */
            real fixedRadiusNN_bruteForce(Tree *tree, Particles *particles, integer *interactions,
                                          integer numParticlesLocal, integer numParticles, integer numNodes);

            /**
             * @brief Wrapper for ::SPH::Kernel::fixedRadiusNN().
             *
             * @return Wall time for kernel execution
             */
            real fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, real radius,
                               integer numParticlesLocal, integer numParticles, integer numNodes);

            /**
             * @brief Wrapper for ::SPH::Kernel::fixedRadiusNN_sharedMemory().
             *
             * @return Wall time for kernel execution
             */
            real fixedRadiusNN_sharedMemory(Tree *tree, Particles *particles, integer *interactions,
                                            integer numParticlesLocal, integer numParticles, integer numNodes);

            /**
             * @brief Wrapper for ::SPH::Kernel::fixedRadiusNN_withinBox().
             *
             * @return Wall time for kernel execution
             */
            real fixedRadiusNN_withinBox(Tree *tree, Particles *particles, integer *interactions,
                                         integer numParticlesLocal, integer numParticles, integer numNodes);

            /**
             * @brief Wrapper for ::SPH::Kernel::fixedRadiusNN_variableSML().
             *
             * @return Wall time for kernel execution
             */
            real fixedRadiusNN_variableSML(Material *materials, Tree *tree, Particles *particles, integer *interactions,
                                           integer numParticlesLocal, integer numParticles,
                                           integer numNodes);

            /**
             * @brief Wrapper for ::SPH::Kernel::compTheta().
             *
             * @return Wall time for kernel execution
             */
            real compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                           DomainList *lowestDomainList, Curve::Type curveType);

            /**
             * @brief Wrapper for ::SPH::Kernel::symbolicForce().
             *
             * @return Wall time for kernel execution
             */
            real symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                              DomainList *lowestDomainList, integer *sendIndices, real searchRadius,
                              integer n, integer m, integer relevantIndex,
                              Curve::Type curveType);

            real symbolicForce_test(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                               DomainList *lowestDomainList, integer *sendIndices, real searchRadius,
                                               integer n, integer m, integer relevantProc, integer relevantIndicesCounter,
                                               integer relevantIndexOld, Curve::Type curveType);

            real symbolicForce_test2(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                     DomainList *domainList, integer *sendIndices, real searchRadius,
                                     integer n, integer m, integer relevantProc, integer relevantIndicesCounter,
                                     Curve::Type curveType);

            /**
             * @brief Wrapper for ::SPH::Kernel::collectSendIndices().
             *
             * @return Wall time for kernel execution
             */
            real collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                               integer *particles2Send, integer *particlesCount,
                                               integer n, integer length, Curve::Type curveType);

            real collectSendIndices_test2(Tree *tree, Particles *particles, integer *sendIndices,
                                          integer *particles2Send, integer *particlesCount,
                                          integer numParticlesLocal, integer numParticles,
                                          integer treeIndex, int currentProc, Curve::Type curveType);

            /**
             * @brief Wrapper for ::SPH::Kernel::particles2Send().
             *
             * @return Wall time for kernel execution
             */
            real particles2Send(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                           DomainList *domainList, DomainList *lowestDomainList, integer maxLevel,
                                           integer *toSend, integer *sendCount, integer *alreadyInserted,
                                           integer insertOffset,
                                           integer numParticlesLocal, integer numParticles, integer numNodes, real radius,
                                           Curve::Type curveType = Curve::lebesgue);

            /**
             * @brief Wrapper for ::SPH::Kernel::collectSendIndicesBackup().
             *
             * @return Wall time for kernel execution
             */
            real collectSendIndicesBackup(integer *toSend, integer *toSendCollected, integer count);

            /**
             * @brief Wrapper for ::SPH::Kernel::collectSendEntriesBackup().
             *
             * @return Wall time for kernel execution
             */
            real collectSendEntriesBackup(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend, integer *sendIndices,
                                    integer *sendCount, integer totalSendCount, integer insertOffset);

            /**
             * @brief Wrapper for ::SPH::Kernel::insertReceivedParticles().
             *
             * @return Wall time for kernel execution
             */
            real insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                         DomainList *domainList, DomainList *lowestDomainList, int n, int m);

            /**
             * @brief Wrapper for ::SPH::Kernel::info().
             *
             * @return Wall time for kernel execution
             */
            real info(Tree *tree, Particles *particles, Helper *helper,
                                 integer numParticlesLocal, integer numParticles, integer numNodes);

            /**
             * @brief Wrapper for ::SPH::Kernel::calculateCentersOfMass().
             *
             * @return Wall time for kernel execution
             */
            real calculateCentersOfMass(Tree *tree, Particles *particles, integer level);

            /**
             * @brief Wrapper for ::SPH::Kernel::determineSearchRadii().
             *
             * @return Wall time for kernel execution
             */
            real determineSearchRadii(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *domainList, DomainList *lowestDomainList, real *searchRadii,
                                      int n, int m, Curve::Type curveType);
        }
    }

}

#endif //MILUPHPC_SPH_CUH
