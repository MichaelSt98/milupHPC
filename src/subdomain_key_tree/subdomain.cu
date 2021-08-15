//
// Created by Michael Staneker on 15.08.21.
//

#include "../../include/subdomain_key_tree/subdomain.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

CUDA_CALLABLE_MEMBER SubDomainKeyTree::SubDomainKeyTree() {

}

CUDA_CALLABLE_MEMBER SubDomainKeyTree::SubDomainKeyTree(integer rank, integer numProcesses, keyType *range) :
                                                        rank(rank), numProcesses(numProcesses), range(range) {

}

CUDA_CALLABLE_MEMBER SubDomainKeyTree::~SubDomainKeyTree() {

}

CUDA_CALLABLE_MEMBER void SubDomainKeyTree::set(integer rank, integer numProcesses, keyType *range) {
    this->rank = rank;
    this->numProcesses = numProcesses;
    this->range = range;
}

namespace SubDomainKeyTreeNS {

    __global__ void setKernel(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range) {
        subDomainKeyTree->set(rank, numProcesses, range);
    }

    void launchSetKernel(SubDomainKeyTree *subDomainKeyTree, integer rank, integer numProcesses, keyType *range) {
        ExecutionPolicy executionPolicy(1, 1);
        cuda::launch(false, executionPolicy, setKernel, subDomainKeyTree, rank, numProcesses, range);
    }

}

CUDA_CALLABLE_MEMBER DomainList::DomainList() {

}

CUDA_CALLABLE_MEMBER DomainList::DomainList(integer *domainListIndices, integer *domainListLevels,
                                            integer *domainListIndex, integer *domainListCounter,
                                            keyType *domainListKeys, keyType *sortedDomainListKeys) :
                                            domainListIndices(domainListIndices), domainListLevels(domainListLevels),
                                            domainListIndex(domainListIndex), domainListCounter(domainListCounter),
                                            domainListKeys(domainListKeys), sortedDomainListKeys(sortedDomainListKeys) {

}

CUDA_CALLABLE_MEMBER DomainList::~DomainList() {

}

CUDA_CALLABLE_MEMBER void DomainList::set(integer *domainListIndices, integer *domainListLevels, integer *domainListIndex,
                              integer *domainListCounter, keyType *domainListKeys, keyType *sortedDomainListKeys) {

    this->domainListIndices = domainListIndices;
    this->domainListLevels = domainListLevels;
    this->domainListIndex = domainListIndex;
    this->domainListCounter = domainListCounter;
    this->domainListKeys = domainListKeys;
    this->sortedDomainListKeys = sortedDomainListKeys;

}

namespace DomainListNS {

    __global__ void setKernel(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                              integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                              keyType *sortedDomainListKeys) {
        domainList->set(domainListIndices, domainListLevels, domainListIndex, domainListCounter, domainListKeys,
                        sortedDomainListKeys);
    }

    void launchSetKernel(DomainList *domainList, integer *domainListIndices, integer *domainListLevels,
                         integer *domainListIndex, integer *domainListCounter, keyType *domainListKeys,
                         keyType *sortedDomainListKeys) {
        ExecutionPolicy executionPolicy(1, 1);
        cuda::launch(false, executionPolicy, setKernel, domainList, domainListIndices, domainListLevels,
                     domainListIndex, domainListCounter, domainListKeys, sortedDomainListKeys);
    }

}