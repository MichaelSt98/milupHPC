#ifndef MILUPHPC_GRAVITY_CUH
#define MILUPHPC_GRAVITY_CUH

#include "../subdomain_key_tree/tree.cuh"
#include "../subdomain_key_tree/subdomain.cuh"
#include "../parameter.h"
#include "../helper.cuh"

namespace Gravity {

    namespace Kernel {

        __global__ void prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                                    Helper *helper, Entry::Name entry);

        __global__ void updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                                    Helper *helper, Entry::Name entry);

        __global__ void compLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList);

        __global__ void compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n);

        __global__ void compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                                      DomainList *lowestDomainList, int n);

        __global__ void computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                      integer warp, integer stackSize);

        __global__ void update(Particles *particles, integer n, real dt, real d);

        __global__ void symbolicForce();

        __global__ void compTheta();

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

        namespace Launch {

            real prepareLowestDomainExchange(Particles *particles, DomainList *lowestDomainList,
                                             Helper *helper, Entry::Name entry);

            real updateLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList,
                                             Helper *helper, Entry::Name entry);

            real compLowestDomainListNodes(Particles *particles, DomainList *lowestDomainList);

            real compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n);

            real compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                               DomainList *lowestDomainList, int n);

            real computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                               integer warp, integer stackSize);

            real update(Particles *particles, integer n, real dt, real d);


            real createKeyHistRanges(Helper *helper, integer bins);

            real keyHistCounter(Tree *tree, Particles *particles, SubDomainKeyTree *subDomainKeyTree,
                                Helper *helper, int bins, int n, Curve::Type curveType=Curve::lebesgue);

            real calculateNewRange(SubDomainKeyTree *subDomainKeyTree, Helper *helper, int bins, int n,
                                   Curve::Type curveType=Curve::lebesgue);

        }
    }
}


#endif //MILUPHPC_GRAVITY_CUH
