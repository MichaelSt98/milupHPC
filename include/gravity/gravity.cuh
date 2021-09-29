#ifndef MILUPHPC_GRAVITY_CUH
#define MILUPHPC_GRAVITY_CUH

#include "../subdomain_key_tree/tree.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "../helper.cuh"
#include <boost/mpi.hpp>

#include <assert.h>

#include <cmath>

namespace Gravity {

    namespace Kernel {

        __global__ void collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                           integer *particles2Send, integer *pseudoParticles2Send,
                                           integer *pseudoParticlesLevel,
                                           integer *particlesCount, integer *pseudoParticlesCount,
                                           integer n, integer length, Curve::Type curveType);

        __global__ void testSendIndices(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                        integer *sendIndices, integer *markedSendIndices,
                                        integer *levels, Curve::Type curveType,
                                        integer length);

        __global__ void zeroDomainListNodes(Particles *particles, DomainList *domainList,
                                            DomainList *lowestDomainList);

        __global__ void prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                                    Helper *helper, Entry::Name entry);

        __global__ void updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                                    Helper *helper, Entry::Name entry);

        __global__ void compLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList);

        __global__ void compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n);

        __global__ void compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                                      DomainList *lowestDomainList, int n);

        __global__ void computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                      integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree);

        __global__ void computeForcesUnsorted(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                       integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree);

        __global__ void computeForcesMiluphcuda(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                    integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree);

        __global__ void update(Particles *particles, integer n, real dt, real d);


        __global__ void intermediateSymbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                  DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                                  integer n, integer m, integer relevantIndex, integer level,
                                                  Curve::Type curveType);

        // Level-wise symbolicForce() to avoid race condition
        __global__ void symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                      integer n, integer m, integer relevantIndex, integer level,
                                      Curve::Type curveType);

        __global__ void compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                  DomainList *domainList, Helper *helper, Curve::Type curveType=Curve::lebesgue);

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

        // Version of inserting received pseudo particles, looping over levels within one kernel
        // problem that __syncthreads() corresponds to blocklevel synchronization!
        /*__global__ void insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                      integer *levels, int n, int m);*/

        __global__ void insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                      integer *levels, int level, int n, int m);

        __global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m);

        __global__ void repairTree(Tree *tree, Particles *particles, DomainList *domainList, int n, int m);

        namespace Launch {

            real collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                    integer *particles2Send, integer *pseudoParticles2Send,
                                    integer *pseudoParticlesLevel,
                                    integer *particlesCount, integer *pseudoParticlesCount,
                                    integer n, integer length, Curve::Type curveType = Curve::lebesgue);

            real testSendIndices(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            integer *sendIndices, integer *markedSendIndices,
                                            integer *levels, Curve::Type curveType,
                                            integer length);

            real zeroDomainListNodes(Particles *particles, DomainList *domainList, DomainList *lowestDomainList);

            real prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                             Helper *helper, Entry::Name entry);

            real updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                             Helper *helper, Entry::Name entry);

            real compLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList);

            real compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n);

            real compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                               DomainList *lowestDomainList, int n);

            real computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                               integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree);

            real computeForcesUnsorted(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                        integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree);

            real computeForcesMiluphcuda(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                        integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree);

            real update(Particles *particles, integer n, real dt, real d);

            real intermediateSymbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                      DomainList *domainList, integer *sendIndices, real diam, real theta_,
                                                      integer n, integer m, integer relevantIndex, integer level,
                                                      Curve::Type curveType);

            real symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                               DomainList *domainList, integer *sendIndices, real diam, real theta_,
                               integer n, integer m, integer relevantIndex, integer level,
                               Curve::Type curveType);

            real compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *domainList, Helper *helper, Curve::Type curveType=Curve::lebesgue);


            real createKeyHistRanges(Helper *helper, integer bins);

            real keyHistCounter(Tree *tree, Particles *particles, SubDomainKeyTree *subDomainKeyTree,
                                Helper *helper, int bins, int n, Curve::Type curveType=Curve::lebesgue);

            real calculateNewRange(SubDomainKeyTree *subDomainKeyTree, Helper *helper, int bins, int n,
                                   Curve::Type curveType=Curve::lebesgue);

            real insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                          integer *levels, int level, int n, int m);

            real insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                    DomainList *domainList, DomainList *lowestDomainList, int n, int m);

            real repairTree(Tree *tree, Particles *particles, DomainList *domainList, int n, int m);

        }
    }
}


#endif //MILUPHPC_GRAVITY_CUH
