//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_DOMAIN_CUH
#define MILUPHPC_DOMAIN_CUH

#include "../parameter.h"
#include "../cuda_utils/cuda_utilities.cuh"

class Key {

};

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

    __global__ void setKernel(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range);
    void launchSetKernel(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range);

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
