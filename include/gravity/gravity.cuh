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
         * The function copies the (pseudo-)particles to contiguous memory as preparation for the communication
         * process via MPI. Particles and pseudo-particles are copied to distinct buffers and are
         * distinguishable by checking the index of the node.
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
         * > The algorithm is similar to the one used in [miluphcuda](https://github.com/christophmschaefer/miluphcuda).
         *
         * To compute gravitational accelerations it is necessary to traverse the tree and check the
         * \f$ \theta \f$-criterion in order to decide which particle-(pseudo-)particle interactions
         * are computed in dependence of $\theta$ and thereby stop the traversal of this sub-tree.
         * Usually this is implemented in a recursive approach which is not efficient for GPUs.
         * This is why an explicit stack is utilized for traversing the tree and computing the gravitational
         * forces in an iterative manner.
         * Shared memory is utilized to remember the current child and node and the cell size in dependence of
         * the tree level or depth in order for being able to calculate the $\theta$-criterion.
         *
         * @note This part of the overall algorithm is one of the most computational intensive ones!
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
         * Refer to ::Gravity::Kernel::computeForces_v1().
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
         * Refer to ::Gravity::Kernel::computeForces_v1().
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
         * > The algorithm is similar to the one used in [An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf).
         *
         * To compute gravitational accelerations it is necessary to traverse the tree and check the
         * \f$ \theta \f$-criterion in order to decide which particle-(pseudo-)particle interactions
         * are computed in dependence of $\theta$ and thereby stop the traversal of this sub-tree.
         * Usually this is implemented in a recursive approach which is not efficient for GPUs.
         * This is why an explicit stack is utilized for traversing the tree and computing the gravitational
         * forces in an iterative manner.
         * Shared memory is used to keep track of the current child and the cell size.
         *
         * @note This part of the overall algorithm is one of the most computational intensive ones!
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
         * Refer to ::Gravity::Kernel::computeForces_v2().
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
         * For more information refer to ::Gravity::Kernel::symbolicForce().
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
         * To avoid race conditions the determination of the (pseudo-)particles
         * (see ::Gravity::Kernel::compTheta()) to be sent is split into two functions which are iteratively
         * and mutually called starting at the root and traversing down to the leaf nodes. An additional array
         * capable of saving an integer for each (pseudo-)particle is initialized with "-1"
         * (do not send this (pseudo-)particle), (pseudo-)particles to be tested with the extended $\theta$-criterion
         * from the extended \f$ \theta \f$-criterion are marked as "0", if they fulfill the condition the "0"
         * is converted to a "3" and the corresponding node's children are marked as "2".
         * The ::Gravity::Kernel::intermediateSymbolicForce() function will convert the "2"s to "0"s preparing
         * the next iteration step, while the "3"s will be converted to a "1" and therefore marked as to be sent.
         *
         * This seems unjustifiably complex but allows for a tree traversing without locks, fences and explicit stack.
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

        __global__ void symbolicForce_test(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                           DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                           integer n, integer m, integer relevantIndex, integer level,
                                           Curve::Type curveType);

        /**
         * @brief Find relevant domain list indices for finding particles to be sent.
         *
         * In order to get the relevant lowest domain list nodes all of them are checked for process assignment and
         * if they do not belong to the executing process their indices are remembered.
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
         * The insertion of the pseudo-particles is basically traversing the local tree and inserting them at the
         * correct position in the tree. Since only whole sub-trees are sent this can be done without any locks
         * and creation of new cells. However, the correct ordering must be fulfilled, hence this kernel is called
         * iteratively and the pseudo-particles are inserted with ascending level, whereas the level information is
         * also communicated via MPI.
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
         * Similar to the pseudo-particle insertion. Since ordering is irrelevant there is no need for an iterative
         * approach and level checking before insertion.
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
             * Refer to ::Gravity::Kernel::computeForces_v1().
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

            real symbolicForce_test(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
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
