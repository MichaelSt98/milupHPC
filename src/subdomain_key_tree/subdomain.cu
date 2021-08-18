//
// Created by Michael Staneker on 15.08.21.
//

#include "../../include/subdomain_key_tree/subdomain.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

__device__ __host__ void KeyNS::key2Char(keyType key, integer maxLevel, char *keyAsChar) {
    int level[21];
    for (int i=0; i<maxLevel; i++) {
        level[i] = (int)(key >> (maxLevel*3 - 3*(i+1)) & (int)7);
    }
    for (int i=0; i<=maxLevel; i++) {
        keyAsChar[2*i] = level[i] + '0';
        keyAsChar[2*i+1] = '|';
    }
    keyAsChar[2*maxLevel+3] = '\0';
}

__device__ __host__ integer KeyNS::key2proc(keyType key, SubDomainKeyTree *s, Curve::Type) {
    //if (curveType == 0) {
    for (int proc=0; proc<s->numProcesses; proc++) {
        if (key >= s->range[proc] && key < s->range[proc+1]) {
            return proc;
        }
    }
    //}
    //else {
    //    unsigned long hilbert = Lebesgue2Hilbert(k, 21);
    //    for (int proc = 0; proc < s->numProcesses; proc++) {
    //        if (hilbert >= s->range[proc] && hilbert < s->range[proc + 1]) {
    //            return proc;
    //        }
    //    }
    //}
    //printf("ERROR: key2proc(k=%lu): -1!", k);
    return -1; // error
}

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