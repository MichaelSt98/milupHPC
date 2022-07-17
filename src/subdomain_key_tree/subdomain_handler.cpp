#include "../../include/subdomain_key_tree/subdomain_handler.h"

SubDomainKeyTreeHandler::SubDomainKeyTreeHandler() {

    boost::mpi::communicator comm;

    h_rank = comm.rank();
    h_numProcesses = comm.size();
    h_range = new keyType[h_numProcesses + 1];
    h_procParticleCounter = new integer[h_numProcesses];

    h_subDomainKeyTree = new SubDomainKeyTree();
    h_subDomainKeyTree->set(h_rank, h_numProcesses, h_range, h_procParticleCounter);

#if TARGET_GPU
    cuda::malloc(d_rank, 1);
    cuda::malloc(d_numProcesses, 1);
    cuda::malloc(d_range, h_numProcesses + 1);
    //cuda::malloc(d_hilberRange, h_numProcesses + 1);
    cuda::malloc(d_procParticleCounter, h_numProcesses);

    cuda::malloc(d_subDomainKeyTree, 1);
    SubDomainKeyTreeNS::Kernel::Launch::set(d_subDomainKeyTree, h_rank, h_numProcesses, d_range, d_procParticleCounter);
#endif // TARGET_GPU

}

SubDomainKeyTreeHandler::~SubDomainKeyTreeHandler() {

    delete [] h_range;
    delete [] h_procParticleCounter;
    delete h_subDomainKeyTree;

#if TARGET_GPU
    cuda::free(d_range);
    cuda::free(d_procParticleCounter);
    cuda::free(d_subDomainKeyTree);
#endif // TARGET_GPU
}

void SubDomainKeyTreeHandler::reset() {
#if TARGET_GPU
    cuda::set(d_procParticleCounter, 0, h_numProcesses);
#endif // TARGET_GPU
    for (int i=0; i<h_numProcesses; i++) {
        h_procParticleCounter[i] = 0;
    }
}

void SubDomainKeyTreeHandler::copy(To::Target target, bool range, bool counter) {
#if TARGET_GPU
    if (range) {
        cuda::copy(h_range, d_range, (h_numProcesses + 1), target);
    }
    if (counter) {
        cuda::copy(h_procParticleCounter, d_procParticleCounter, h_numProcesses, target);
    }
#endif
}

DomainListHandler::DomainListHandler(integer domainListSize) : domainListSize(domainListSize) {

    h_domainListIndices = new integer[domainListSize];
    h_domainListLevels = new integer[domainListSize];
    h_domainListIndex = new integer;
    h_domainListCounter = new integer;
    h_domainListKeys = new keyType [domainListSize];
    h_sortedDomainListKeys = new keyType[domainListSize];
    h_relevantDomainListIndices = new integer[domainListSize];
    h_relevantDomainListLevels = new integer[domainListSize];
    h_relevantDomainListProcess = new integer[domainListSize];
    h_relevantDomainListOriginalIndex = new integer[domainListSize];

    h_borders = new real[2 * DIM * domainListSize];

    h_domainList = new DomainList;
    h_domainList->set(h_domainListIndices, h_domainListLevels,
                      h_domainListIndex, h_domainListCounter, h_domainListKeys,
                      h_sortedDomainListKeys, h_relevantDomainListIndices,
                      h_relevantDomainListLevels, h_relevantDomainListProcess);

    h_domainList->setBorders(h_borders, h_relevantDomainListOriginalIndex);

#if TARGET_GPU
    cuda::malloc(d_domainListIndices, domainListSize);
    cuda::malloc(d_domainListLevels, domainListSize);
    cuda::malloc(d_domainListIndex, 1);
    cuda::malloc(d_domainListCounter, 1);
    cuda::malloc(d_domainListKeys, domainListSize);
    cuda::malloc(d_sortedDomainListKeys, domainListSize);
    cuda::malloc(d_relevantDomainListIndices, domainListSize);
    cuda::malloc(d_relevantDomainListLevels, domainListSize);
    cuda::malloc(d_relevantDomainListProcess, domainListSize);
    cuda::malloc(d_relevantDomainListOriginalIndex, domainListSize);

    cuda::malloc(d_borders, 2 * DIM * domainListSize);

    cuda::malloc(d_domainList, 1);
    DomainListNS::Kernel::Launch::set(d_domainList, d_domainListIndices, d_domainListLevels,
                                      d_domainListIndex, d_domainListCounter, d_domainListKeys,
                                      d_sortedDomainListKeys, d_relevantDomainListIndices,
                                      d_relevantDomainListLevels, d_relevantDomainListProcess);
    DomainListNS::Kernel::Launch::setBorders(d_domainList, d_borders, d_relevantDomainListOriginalIndex);
#endif

}

DomainListHandler::~DomainListHandler() {

    delete [] h_domainListIndices;
    delete [] h_domainListLevels;
    delete h_domainListIndex;
    delete h_domainListCounter;
    delete [] h_domainListKeys;
    delete [] h_sortedDomainListKeys;
    delete [] h_relevantDomainListIndices;
    delete [] h_relevantDomainListLevels;
    delete [] h_relevantDomainListProcess;
    delete [] h_relevantDomainListOriginalIndex;
    delete [] h_borders;
    delete h_domainList;

#if TARGET_GPU
   cuda::free(d_domainListIndices);
   cuda::free(d_domainListLevels);
   cuda::free(d_domainListIndex);
   cuda::free(d_domainListCounter);
   cuda::free(d_domainListKeys);
   cuda::free(d_sortedDomainListKeys);
   cuda::free(d_relevantDomainListIndices);
   cuda::free(d_relevantDomainListLevels);
   cuda::free(d_relevantDomainListProcess);
   cuda::free(d_relevantDomainListOriginalIndex);
   cuda::free(d_borders);
   cuda::free(d_domainList);
#endif

}

void DomainListHandler::reset() {

    keyType maxKey = (keyType)KEY_MAX;
    std::fill(h_domainListIndices, h_domainListIndices + domainListSize, -1);
    std::fill(h_domainListLevels, h_domainListLevels + domainListSize, -1);
    h_domainListIndex = 0;
    h_domainListCounter = 0;
    std::fill(h_domainListKeys, h_domainListKeys + domainListSize, maxKey);
    std::fill(h_sortedDomainListKeys, h_sortedDomainListKeys + domainListSize, maxKey);
    std::fill(h_relevantDomainListIndices, h_relevantDomainListIndices + domainListSize, -1);

#if TARGET_GPU
    cuda::set(d_domainListIndices, -1, domainListSize);
    cuda::set(d_domainListLevels, -1, domainListSize);
    cuda::set(d_domainListIndex, 0, 1);
    cuda::set(d_domainListCounter, 0, 1);
    cuda::set(d_domainListKeys, maxKey, domainListSize);
    cuda::set(d_sortedDomainListKeys, maxKey, domainListSize);
    cuda::set(d_relevantDomainListIndices, -1, domainListSize);
#endif
}

