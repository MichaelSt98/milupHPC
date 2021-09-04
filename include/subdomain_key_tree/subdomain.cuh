//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_DOMAIN_CUH
#define MILUPHPC_DOMAIN_CUH

#include "../parameter.h"
#include "../cuda_utils/cuda_utilities.cuh"
#include "tree.cuh"

//class Tree;
//class Particles;
class DomainList;

class Key {

    /*keyType *keys;
    integer *maxLevel;

    CUDA_CALLABLE_MEMBER Key();
    CUDA_CALLABLE_MEMBER Key(keyType *keys, integer *maxLevel);
    CUDA_CALLABLE_MEMBER ~Key();
    CUDA_CALLABLE_MEMBER void set(keyType *keys, integer *maxLevel);*/

};

class SubDomainKeyTree;

namespace KeyNS {

    CUDA_CALLABLE_MEMBER void key2Char(keyType key, integer maxLevel, char *keyAsChar);
    CUDA_CALLABLE_MEMBER integer key2proc(keyType key, SubDomainKeyTree *subDomainKeyTree/*, Curve::Type curveType=Curve::lebesgue*/);
}

class SubDomainKeyTree {

public:
    integer rank;
    integer numProcesses;
    keyType *range;

    integer *procParticleCounter;

    CUDA_CALLABLE_MEMBER SubDomainKeyTree();
    CUDA_CALLABLE_MEMBER SubDomainKeyTree(integer rank, integer numProcesses, keyType *range,
                                          integer *procParticleCounter);
    CUDA_CALLABLE_MEMBER ~SubDomainKeyTree();
    CUDA_CALLABLE_MEMBER void set(integer rank, integer numProcesses, keyType *range, integer *procParticleCounter);
    CUDA_CALLABLE_MEMBER integer key2proc(keyType key/*, Curve::Type curveType=Curve::lebesgue*/);
    CUDA_CALLABLE_MEMBER bool isDomainListNode(keyType key, integer maxLevel, integer level,
                                               Curve::Type curveType=Curve::lebesgue);

};

namespace SubDomainKeyTreeNS {

    namespace Kernel {

        __global__ void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range,
                            integer *procParticleCounter);

        __global__ void test(SubDomainKeyTree *subDomainKeyTree);

        __global__ void buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m);

        __global__ void getParticleKeys(SubDomainKeyTree *subDomainKeyTree, Tree *tree,
                                                        Particles *particles, keyType *keys, integer maxLevel,
                                                        integer n, Curve::Type curveType = Curve::lebesgue);

        __global__ void particlesPerProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                            integer n, integer m, Curve::Type curveType=Curve::lebesgue);

        __global__ void markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                             integer n, integer m, integer *sortArray,
                                             Curve::Type curveType=Curve::lebesgue);

        namespace Launch {

            void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range,
                     integer *procParticleCounter);

            void test(SubDomainKeyTree *subDomainKeyTree);

            real buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m);

            real getParticleKeys(SubDomainKeyTree *subDomainKeyTree, Tree *tree,
                                 Particles *particles, keyType *keys, integer maxLevel,
                                 integer n, Curve::Type curveType = Curve::lebesgue);

            real particlesPerProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                integer n, integer m, Curve::Type curveType=Curve::lebesgue);

            real markParticlesProcess(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                 integer n, integer m, integer *sortArray,
                                                 Curve::Type curveType=Curve::lebesgue);
        }

    }

}

class DomainList {

public:

    integer *domainListIndices;
    integer *domainListLevels;
    integer *domainListIndex;
    integer *domainListCounter;
    keyType *domainListKeys;
    keyType *sortedDomainListKeys;
    integer *relevantDomainListIndices;

    CUDA_CALLABLE_MEMBER DomainList();
    CUDA_CALLABLE_MEMBER DomainList(integer *domainListIndices, integer *domainListLevels, integer *domainListIndex,
                                    integer *domainListCounter, keyType *domainListKeys, keyType *sortedDomainListKeys,
                                    integer *relevantDomainListIndices);
    CUDA_CALLABLE_MEMBER ~DomainList();
    CUDA_CALLABLE_MEMBER void set(integer *domainListIndices, integer *domainListLevels, integer *domainListIndex,
                                  integer *domainListCounter, keyType *domainListKeys, keyType *sortedDomainListKeys,
                                  integer *relevantDomainListIndices);

};

namespace DomainListNS {

    namespace Kernel {
        __global__ void set(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                            integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                            keyType *sortedDomainListKeys, integer *relevantDomainListIndices);

        __global__ void info(Particles *particles, DomainList *domainList);

        __global__ void info(Particles *particles, DomainList *domainList, DomainList *lowestDomainList);

        __global__ void createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                         integer maxLevel, Curve::Type curveType = Curve::lebesgue);

        __global__ void lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, DomainList *domainList,
                                         DomainList *lowestDomainList, integer n, integer m);

        namespace Launch {
            void set(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                     integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                     keyType *sortedDomainListKeys, integer *relevantDomainListIndices);

            real info(Particles *particles, DomainList *domainList);

            real info(Particles *particles, DomainList *domainList, DomainList *lowestDomainList);

            real createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                  integer maxLevel, Curve::Type curveType = Curve::lebesgue);

            real lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, DomainList *domainList,
                                             DomainList *lowestDomainList, integer n, integer m);
        }
    }

}

#endif //MILUPHPC_DOMAIN_CUH
