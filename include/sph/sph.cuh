#ifndef MILUPHPC_SPH_CUH
#define MILUPHPC_SPH_CUH

#include "../particles.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "../helper.cuh"

#include <boost/mpi.hpp>
#include <assert.h>

namespace SPH {

    void exchangeParticleEntry(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend, integer *sendLengths,
                               integer *receiveLengths, integer numParticlesLocal);

    namespace Kernel {

        __global__ void nearNeighbourSearch(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                                            integer numParticles, integer numNodes);
        __global__ void
        fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                      integer numParticles, integer numNodes);

        __global__ void
        fixedRadiusNN_Test(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                                    integer numParticles, integer numNodes);

        __global__ void compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                  DomainList *lowestDomainList, Curve::Type curveType);

        __global__ void symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                      DomainList *lowestDomainList, integer *sendIndices,
                                      integer n, integer m, integer relevantIndex,
                                      Curve::Type curveType);

        __global__ void collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                           integer *particles2Send, integer *particlesCount,
                                           integer n, integer length, Curve::Type curveType);

        __global__ void particles2Send(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                       DomainList *domainList, DomainList *lowestDomainList, integer maxLevel,
                                       integer *toSend, integer *sendCount, integer *alreadyInserted,
                                       integer insertOffset,
                                       integer numParticlesLocal, integer numParticles, integer numNodes, real radius,
                                       Curve::Type curveType = Curve::lebesgue);

        __global__ void collectSendIndicesBackup(integer *toSend, integer *toSendCollected, integer count);

        __global__ void collectSendEntriesBackup(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend,
                                           integer *sendIndices, integer *sendCount, integer totalSendCount,
                                           integer insertOffset);

        __global__ void insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                DomainList *domainList, DomainList *lowestDomainList, int n, int m);

        __global__ void info(Tree *tree, Particles *particles, Helper *helper,
                             integer numParticlesLocal, integer numParticles, integer numNodes);

        namespace Launch {

            real fixedRadiusNN(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                          integer numParticles, integer numNodes);

            real fixedRadiusNN_Test(Tree *tree, Particles *particles, integer *interactions, integer numParticlesLocal,
                               integer numParticles, integer numNodes);

            real compTheta(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                           DomainList *lowestDomainList, Curve::Type curveType);

            real symbolicForce(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                              DomainList *lowestDomainList, integer *sendIndices,
                              integer n, integer m, integer relevantIndex,
                              Curve::Type curveType);

            real collectSendIndices(Tree *tree, Particles *particles, integer *sendIndices,
                                               integer *particles2Send, integer *particlesCount,
                                               integer n, integer length, Curve::Type curveType);

            real particles2Send(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                           DomainList *domainList, DomainList *lowestDomainList, integer maxLevel,
                                           integer *toSend, integer *sendCount, integer *alreadyInserted,
                                           integer insertOffset,
                                           integer numParticlesLocal, integer numParticles, integer numNodes, real radius,
                                           Curve::Type curveType = Curve::lebesgue);

            real collectSendIndicesBackup(integer *toSend, integer *toSendCollected, integer count);

            real collectSendEntriesBackup(SubDomainKeyTree *subDomainKeyTree, real *entry, real *toSend, integer *sendIndices,
                                    integer *sendCount, integer totalSendCount, integer insertOffset);

            real insertReceivedParticles(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                         DomainList *domainList, DomainList *lowestDomainList, int n, int m);

            real info(Tree *tree, Particles *particles, Helper *helper,
                                 integer numParticlesLocal, integer numParticles, integer numNodes);
        }
    }

}

#endif //MILUPHPC_SPH_CUH
