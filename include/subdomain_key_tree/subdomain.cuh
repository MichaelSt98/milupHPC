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

    //TODO: 1D and 2D DirTable
    // table needed to convert from Lebesgue to Hilbert keys
    CUDA_CALLABLE_MEMBER const unsigned char DirTable[12][8] =
            { { 8,10, 3, 3, 4, 5, 4, 5}, { 2, 2,11, 9, 4, 5, 4, 5},
              { 7, 6, 7, 6, 8,10, 1, 1}, { 7, 6, 7, 6, 0, 0,11, 9},
              { 0, 8, 1,11, 6, 8, 6,11}, {10, 0, 9, 1,10, 7, 9, 7},
              {10, 4, 9, 4,10, 2, 9, 3}, { 5, 8, 5,11, 2, 8, 3,11},
              { 4, 9, 0, 0, 7, 9, 2, 2}, { 1, 1, 8, 5, 3, 3, 8, 6},
              {11, 5, 0, 0,11, 6, 2, 2}, { 1, 1, 4,10, 3, 3, 7,10} };

    //TODO: 1D and 2d Hilbert table
    // table needed to convert from Lebesgue to Hilbert keys
    CUDA_CALLABLE_MEMBER const unsigned char HilbertTable[12][8] = { {0,7,3,4,1,6,2,5}, {4,3,7,0,5,2,6,1}, {6,1,5,2,7,0,4,3},
                                                           {2,5,1,6,3,4,0,7}, {0,1,7,6,3,2,4,5}, {6,7,1,0,5,4,2,3},
                                                           {2,3,5,4,1,0,6,7}, {4,5,3,2,7,6,0,1}, {0,3,1,2,7,4,6,5},
                                                           {2,1,3,0,5,6,4,7}, {4,7,5,6,3,0,2,1}, {6,5,7,4,1,2,0,3} };

    CUDA_CALLABLE_MEMBER void key2Char(keyType key, integer maxLevel, char *keyAsChar);
    CUDA_CALLABLE_MEMBER integer key2proc(keyType key, SubDomainKeyTree *subDomainKeyTree,
                                          Curve::Type curveType=Curve::lebesgue);

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
    CUDA_CALLABLE_MEMBER integer key2proc(keyType key, Curve::Type curveType=Curve::lebesgue);
    CUDA_CALLABLE_MEMBER bool isDomainListNode(keyType key, integer maxLevel, integer level,
                                               Curve::Type curveType=Curve::lebesgue);

};

namespace SubDomainKeyTreeNS {

    namespace Kernel {

        __global__ void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range,
                            integer *procParticleCounter);

        __global__ void test(SubDomainKeyTree *subDomainKeyTree);

        __global__ void buildDomainTree(Tree *tree, Particles *particles, DomainList *domainList, integer n, integer m);

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


        __global__ void createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                         integer maxLevel, Curve::Type curveType = Curve::lebesgue);

        __global__ void lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, DomainList *domainList,
                                         DomainList *lowestDomainList, integer n, integer m);

        namespace Launch {
            void set(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                     integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                     keyType *sortedDomainListKeys, integer *relevantDomainListIndices);

            real createDomainList(SubDomainKeyTree *subDomainKeyTree, DomainList *domainList,
                                  integer maxLevel, Curve::Type curveType = Curve::lebesgue);

            real lowestDomainList(SubDomainKeyTree *subDomainKeyTree, Tree *tree, DomainList *domainList,
                                             DomainList *lowestDomainList, integer n, integer m);
        }
    }

}

#endif //MILUPHPC_DOMAIN_CUH
