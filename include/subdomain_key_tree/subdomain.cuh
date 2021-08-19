//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_DOMAIN_CUH
#define MILUPHPC_DOMAIN_CUH

#include "../parameter.h"
#include "../cuda_utils/cuda_utilities.cuh"

class Tree;
class Particles;

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

    //TODO: 2D DirTable
    // table needed to convert from Lebesgue to Hilbert keys
    __device__ const unsigned char DirTable[12][8] =
            { { 8,10, 3, 3, 4, 5, 4, 5}, { 2, 2,11, 9, 4, 5, 4, 5},
              { 7, 6, 7, 6, 8,10, 1, 1}, { 7, 6, 7, 6, 0, 0,11, 9},
              { 0, 8, 1,11, 6, 8, 6,11}, {10, 0, 9, 1,10, 7, 9, 7},
              {10, 4, 9, 4,10, 2, 9, 3}, { 5, 8, 5,11, 2, 8, 3,11},
              { 4, 9, 0, 0, 7, 9, 2, 2}, { 1, 1, 8, 5, 3, 3, 8, 6},
              {11, 5, 0, 0,11, 6, 2, 2}, { 1, 1, 4,10, 3, 3, 7,10} };

    //TODO: 2d Hilbert table
    // table needed to convert from Lebesgue to Hilbert keys
    __device__ const unsigned char HilbertTable[12][8] = { {0,7,3,4,1,6,2,5}, {4,3,7,0,5,2,6,1}, {6,1,5,2,7,0,4,3},
                                                           {2,5,1,6,3,4,0,7}, {0,1,7,6,3,2,4,5}, {6,7,1,0,5,4,2,3},
                                                           {2,3,5,4,1,0,6,7}, {4,5,3,2,7,6,0,1}, {0,3,1,2,7,4,6,5},
                                                           {2,1,3,0,5,6,4,7}, {4,7,5,6,3,0,2,1}, {6,5,7,4,1,2,0,3} };

    __device__ __host__ void key2Char(keyType key, integer maxLevel, char *keyAsChar);
    __device__ __host__ integer key2proc(keyType key, SubDomainKeyTree *s, Curve::Type=Curve::lebesgue);

}

class SubDomainKeyTree {

public:
    integer rank;
    integer numProcesses;
    keyType *range;

    CUDA_CALLABLE_MEMBER SubDomainKeyTree();
    CUDA_CALLABLE_MEMBER SubDomainKeyTree(integer rank, integer numProcesses, keyType *range);
    CUDA_CALLABLE_MEMBER ~SubDomainKeyTree();
    CUDA_CALLABLE_MEMBER void set(integer rank, integer numProcesses, keyType *range);

};

namespace SubDomainKeyTreeNS {

    namespace Kernel {

        __global__ void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range);

        __global__ void test(SubDomainKeyTree *subDomainKeyTree);

        namespace Launch {

            void set(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range);

            void test(SubDomainKeyTree *subDomainKeyTree);
        }

        __global__ void particlesPerProcessKernel(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                  Curve::Type = Curve::lebesgue);

        __global__ void markParticlesProcessKernel(SubDomainKeyTree *subDomainKeyTree, Tree *tree, Particles *particles,
                                                   integer *sortArray, Curve::Type = Curve::lebesgue);

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

    CUDA_CALLABLE_MEMBER DomainList();
    CUDA_CALLABLE_MEMBER DomainList(integer *domainListIndices, integer *domainListLevels, integer *domainListIndex,
                                    integer *domainListCounter, keyType *domainListKeys, keyType *sortedDomainListKeys);
    CUDA_CALLABLE_MEMBER ~DomainList();
    CUDA_CALLABLE_MEMBER void set(integer *domainListIndices, integer *domainListLevels, integer *domainListIndex,
                                  integer *domainListCounter, keyType *domainListKeys, keyType *sortedDomainListKeys);

};

namespace DomainListNS {

    __global__ void setKernel(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                              integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                              keyType *sortedDomainListKeys);

    void launchSetKernel(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                         integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                         keyType *sortedDomainListKeys);

}

#endif //MILUPHPC_DOMAIN_CUH
