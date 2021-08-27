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

        __global__ void compLowestDomainListNodesKernel(Particles *particles, DomainList *lowestDomainList);

        __global__ void compLocalPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList, int n);

        __global__ void compDomainListPseudoParticles(Tree *tree, Particles *particles, DomainList *domainList,
                                                      DomainList *lowestDomainList, int n);

        __global__ void computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                                      integer warp, integer stackSize);

        __global__ void symbolicForce();

        __global__ void compTheta();

        namespace Launch {
            real computeForces(Tree *tree, Particles *particles, integer n, integer m, integer blockSize,
                               integer warp, integer stackSize);
        }
    }
}


#endif //MILUPHPC_GRAVITY_CUH
