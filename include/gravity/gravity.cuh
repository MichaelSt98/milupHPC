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

        __global__ void computeForces_v1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                         SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing);

        __global__ void computeForces_v1_1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                         SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing);

        __global__ void computeForces_v1_2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                           SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing);

        __global__ void computeForces_v2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                         integer blockSize, integer warp, integer stackSize,
                                         SubDomainKeyTree *subDomainKeyTree, real theta,
                                         real smoothing);

        __global__ void computeForces_v2_1(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                           integer warp, integer stackSize, SubDomainKeyTree *subDomainKeyTree,
                                           real theta, real smoothing);

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

        // Version of inserting received pseudo particles, looping over levels within one kernel
        // problem that __syncthreads() corresponds to blocklevel synchronization!
        /*__global__ void insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                      integer *levels, int n, int m);*/

        __global__ void insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                      integer *levels, int level, int n, int m);

        __global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m);


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

            real computeForces_v1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                  SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing);

            real computeForces_v1_1(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                  SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing);

            real computeForces_v1_2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                    SubDomainKeyTree *subDomainKeyTree, real theta, real smoothing);

            real computeForces_v2(Tree *tree, Particles *particles, real radius, integer n, integer m,
                                  integer blockSize, integer warp, integer stackSize,
                                  SubDomainKeyTree *subDomainKeyTree, real theta,
                                  real smoothing);

            real computeForces_v2_1(Tree *tree, Particles *particles, integer n, integer m,
                                    integer blockSize, integer warp, integer stackSize,
                                    SubDomainKeyTree *subDomainKeyTree, real theta,
                                    real smoothing);

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

            real insertReceivedPseudoParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                          integer *levels, int level, int n, int m);

            real insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                    DomainList *domainList, DomainList *lowestDomainList, int n, int m);

        }
    }
}


#endif //MILUPHPC_GRAVITY_CUH
