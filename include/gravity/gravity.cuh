/**
 * @file gravity.cuh
 * @brief Gravitational force computation functions and CUDA kernels.
 *
 * Gravitational force computation functions and CUDA kernels for computing the gravitational
 * forces according to the Barnes-Hut method.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_GRAVITY_CUH
#define MILUPHPC_GRAVITY_CUH

#include "../subdomain_key_tree/tree.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "../helper.cuh"
#include <boost/mpi.hpp>
#include <assert.h>
#include <cmath>

/// Gravity related kernels/functions.
namespace Gravity {

    /// CUDA kernel functions.
    namespace Kernel {

        /**
         * @brief Collect the send indices.
         *
         * Collect the previous marked indices to be sent by copying to contiguous memory in order to send the
         * particles or rather particle entries using MPI.
         *
         * @param tree Tree class instance.
         * @param particles Particles class instance.
         * @param sendIndices Marked as to be sent or not.
         * @param[out] particles2Send Particles to be sent.
         * @param[out] pseudoParticles2Send Pseudo-particles to be sent.
         * @param[out] pseudoParticlesLevel Pseudo-particles level(s) (to be sent).
         * @param particlesCount Particles to be sent counter.
         * @param pseudoParticlesCount Pseudo-particles to be sent counter.
         * @param n
         * @param length
         * @param curveType Space-filling curve type.
         */
        __global__ void collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                           integer *particles2Send, integer *pseudoParticles2Send,
                                           integer *pseudoParticlesLevel,
                                           integer *particlesCount, integer *pseudoParticlesCount,
                                           integer n, integer length, Curve::Type curveType);

        /**
         * @brief Test the send indices.
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance.
         * @param tree Tree class instance.
         * @param particles Particles class instance.
         * @param sendIndices
         * @param markedSendIndices
         * @param levels
         * @param curveType
         * @param length
         */
        __global__ void testSendIndices(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                        integer *sendIndices, integer *markedSendIndices,
                                        integer *levels, Curve::Type curveType,
                                        integer length);

        /**
         * @brief Compute gravitational forces according to Barnes-Hut method.
         *
         * @param tree Tree class instance.
         * @param particles Particles class instance.
         * @param radius
         * @param n
         * @param m
         * @param subDomainKeyTree SubDomainKeyTree instance.
         * @param theta Clumping parameter/\f$ \theta $\f parameter
         * @param smoothing Gravitational smoothing parameter.
         * @param potentialEnergy Calculate potential energy.
         */
        __global__ void computeForces_v1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                         SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing,
                                         bool potentialEnergy=false);

        /**
         * @brief Compute gravitational forces according to Barnes-Hut method.
         *
         * @param tree Tree class instance.
         * @param particles Particles class instance.
         * @param radius
         * @param n
         * @param m
         * @param subDomainKeyTree
         * @param theta
         * @param smoothing
         * @param potentialEnergy
         */
        __global__ void computeForces_v1_1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                         SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing,
                                         bool potentialEnergy=false);

        /**
         * @brief Compute gravitational forces according to Barnes-Hut method.
         *
         * @param tree Tree class instance.
         * @param particles Particles class instance.
         * @param radius
         * @param n
         * @param m
         * @param subDomainKeyTree
         * @param theta
         * @param smoothing
         * @param potentialEnergy
         */
        __global__ void computeForces_v1_2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                           SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing,
                                           bool potentialEnergy=false);

        /**
         * @brief Compute gravitational forces according to Barnes-Hut method.
         *
         * @param tree Tree class instance.
         * @param particles Particles class instance.
         * @param radius
         * @param n
         * @param m
         * @param blockSize
         * @param warp
         * @param stackSize
         * @param subDomainKeyTree
         * @param theta
         * @param smoothing
         * @param potentialEnergy
         */
        __global__ void computeForces_v2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                         integer blockSize, integer warp, integer stackSize,
                                         SubDomainKeyTree *subDomainKeyTree, real theta,
                                         real smoothing, bool potentialEnergy=false);

        /**
         * @brief Compute gravitational forces according to Barnes-Hut method.
         *
         * @param tree Tree class instance.
         * @param particles Particles class instance.
         * @param n
         * @param m
         * @param blockSize
         * @param warp
         * @param stackSize
         * @param subDomainKeyTree
         * @param theta
         * @param smoothing
         * @param potentialEnergy
         */
        __global__ void computeForces_v2_1(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                           integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree,
                                           real theta, real smoothing, bool potentialEnergy=false);

        /**
         * @brief Find particles to be sent (part 1).
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance.
         * @param tree Tree class instance.
         * @param particles
         * @param domainList
         * @param sendIndices
         * @param diam
         * @param theta_
         * @param n
         * @param m
         * @param relevantIndex
         * @param level
         * @param curveType
         */
        __global__ void intermediateSymbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                  DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                                  integer n, integer m, integer relevantIndex, integer level,
                                                  Curve::Type curveType);

        // Level-wise symbolicForce() to avoid race condition
        /**
         * @brief Find particles to be sent (part 2).
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance.
         * @param tree Tree class instance.
         * @param particles
         * @param domainList
         * @param sendIndices
         * @param diam
         * @param theta_
         * @param n
         * @param m
         * @param relevantIndex
         * @param level
         * @param curveType
         */
        __global__ void symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                      integer n, integer m, integer relevantIndex, integer level,
                                      Curve::Type curveType);

        /**
         * @brief Find relevant domain list indices for finding particles to be sent.
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance.
         * @param tree
         * @param particles
         * @param domainList
         * @param helper
         * @param curveType
         */
        __global__ void compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                  DomainList *domainList, Helper *helper, Curve::Type curveType=Curve::lebesgue);

        // Version of inserting received pseudo particles, looping over levels within one kernel
        // problem that __syncthreads() corresponds to blocklevel synchronization!
        /*__global__ void insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                      integer *levels, int n, int m);*/

        /**
         * @brief Insert received pseudo-particles.
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance.
         * @param tree
         * @param particles
         * @param levels
         * @param level
         * @param n
         * @param m
         */
        __global__ void insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                      integer *levels, int level, int n, int m);

        /**
         * @brief Insert received particles.
         *
         * @param subDomainKeyTree SubDomainKeyTree class instance.
         * @param tree
         * @param particles
         * @param domainList
         * @param lowestDomainList
         * @param n
         * @param m
         */
        __global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m);


        /// Wrapper for CUDA kernels/Enable to launch them from cpp file.
        namespace Launch {

            /**
             * @brief Wrapper for: Gravity::Kernel::collectSendIndices().
             *
             * @return Wall time for kernel execution.
             */
            real collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                    integer *particles2Send, integer *pseudoParticles2Send,
                                    integer *pseudoParticlesLevel,
                                    integer *particlesCount, integer *pseudoParticlesCount,
                                    integer n, integer length, Curve::Type curveType = Curve::lebesgue);

            /**
             * @brief Wrapper for: Gravity::Kernel::testSendIndices().
             *
             * @return Wall time for kernel execution.
             */
            real testSendIndices(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            integer *sendIndices, integer *markedSendIndices,
                                            integer *levels, Curve::Type curveType,
                                            integer length);

            /**
             * @brief Wrapper for: Gravity::Kernel::computeForces_v1().
             *
             * @return Wall time for kernel execution.
             */
            real computeForces_v1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                  SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing,
                                  bool potentialEnergy=false);

            /**
             * @brief Wrapper for: Gravity::Kernel::computeForces_v1_1().
             *
             * @return Wall time for kernel execution.
             */
            real computeForces_v1_1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                  SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing,
                                    bool potentialEnergy=false);

            /**
             * @brief Wrapper for: Gravity::Kernel::computeForces_v1_2().
             *
             * @return Wall time for kernel execution.
             */
            real computeForces_v1_2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                    SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing,
                                    bool potentialEnergy=false);

            /**
             * @brief Wrapper for: Gravity::Kernel::computeForces_v2().
             *
             * @return Wall time for kernel execution.
             */
            real computeForces_v2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                  integer blockSize, integer warp, integer stackSize,
                                  SubDomainKeyTree *subDomainKeyTree, real theta,
                                  real smoothing, bool potentialEnergy=false);

            /**
             * @brief Wrapper for: Gravity::Kernel::computeForces_v2_1().
             *
             * @return Wall time for kernel execution.
             */
            real computeForces_v2_1(Tree *tree, Particles *particles, integer n, integer m,
                                    integer blockSize, integer warp, integer stackSize,
                                    SubDomainKeyTree *subDomainKeyTree, real theta,
                                    real smoothing, bool potentialEnergy=false);

            /**
             * @brief Wrapper for: Gravity::Kernel::intermediateSymbolicForce().
             *
             * @return Wall time for kernel execution.
             */
            real intermediateSymbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                      DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                                      integer n, integer m, integer relevantIndex, integer level,
                                                      Curve::Type curveType);

            /**
             * @brief Wrapper for: Gravity::Kernel::symbolicForce().
             *
             * @return Wall time for kernel execution.
             */
            real symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                               DomainList *domainList, integer *sendIndices, real diam, real theta_,
                               integer n, integer m, integer relevantIndex, integer level,
                               Curve::Type curveType);

            /**
             * @brief Wrapper for: Gravity::Kernel::compTheta().
             *
             * @return Wall time for kernel execution.
             */
            real compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *domainList, Helper *helper, Curve::Type curveType=Curve::lebesgue);

            /**
             * @brief Wrapper for: Gravity::Kernel::insertReceivedPseudoParticles().
             *
             * @return Wall time for kernel execution.
             */
            real insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                          integer *levels, int level, int n, int m);

            /**
             * @brief Wrapper for: Gravity::Kernel::insertReceivedParticles().
             *
             * @return Wall time for kernel execution.
             */
            real insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                    DomainList *domainList, DomainList *lowestDomainList, int n, int m);

        }
    }
}


#endif //MILUPHPC_GRAVITY_CUH
