#include "../../include/subdomain_key_tree/subdomain_handler.h"

SubDomainKeyTreeHandler::SubDomainKeyTreeHandler() {

    boost::mpi::communicator comm;

    h_rank = comm.rank();
    h_numProcesses = comm.size();
    h_range = new keyType[h_numProcesses + 1];
    h_procParticleCounter = new integer[h_numProcesses];

    h_subDomainKeyTree = new SubDomainKeyTree();
    h_subDomainKeyTree->set(h_rank, h_numProcesses, h_range, h_procParticleCounter);

    cuda::malloc(d_rank, 1);
    cuda::malloc(d_numProcesses, 1);
    cuda::malloc(d_range, h_numProcesses + 1);
    cuda::malloc(d_procParticleCounter, h_numProcesses);

    cuda::malloc(d_subDomainKeyTree, 1);
    SubDomainKeyTreeNS::Kernel::Launch::set(d_subDomainKeyTree, h_rank, h_numProcesses, d_range, d_procParticleCounter);
}

SubDomainKeyTreeHandler::~SubDomainKeyTreeHandler() {

    delete [] h_range;
    delete [] h_procParticleCounter;
    delete h_subDomainKeyTree;

   cuda::free(d_range);
   cuda::free(d_procParticleCounter);
   cuda::free(d_subDomainKeyTree);
}

void SubDomainKeyTreeHandler::reset() {
    cuda::set(d_procParticleCounter, 0, h_numProcesses);
    for (int i=0; i<h_numProcesses; i++) {
        h_procParticleCounter[i] = 0;
    }
}

void SubDomainKeyTreeHandler::copy(To::Target target, bool range, bool counter) {
    if (range) {
        cuda::copy(h_range, d_range, (h_numProcesses + 1), target);
    }
    if (counter) {
        cuda::copy(h_procParticleCounter, d_procParticleCounter, h_numProcesses, target);
    }
}

DomainListHandler::DomainListHandler(integer domainListSize) : domainListSize(domainListSize) {

    cuda::malloc(d_domainListIndices, domainListSize);
    cuda::malloc(d_domainListLevels, domainListSize);
    cuda::malloc(d_domainListIndex, 1);
    cuda::malloc(d_domainListCounter, 1);
    cuda::malloc(d_domainListKeys, domainListSize);
    cuda::malloc(d_sortedDomainListKeys, domainListSize);
    cuda::malloc(d_relevantDomainListIndices, domainListSize);
    cuda::malloc(d_relevantDomainListProcess, domainListSize);

    cuda::malloc(d_domainList, 1);
    DomainListNS::Kernel::Launch::set(d_domainList, d_domainListIndices, d_domainListLevels,
                                      d_domainListIndex, d_domainListCounter, d_domainListKeys,
                                      d_sortedDomainListKeys, d_relevantDomainListIndices,
                                      d_relevantDomainListProcess);
}

DomainListHandler::~DomainListHandler() {

   cuda::free(d_domainListIndices);
   cuda::free(d_domainListLevels);
   cuda::free(d_domainListIndex);
   cuda::free(d_domainListCounter);
   cuda::free(d_domainListKeys);
   cuda::free(d_sortedDomainListKeys);
   cuda::free(d_relevantDomainListIndices);
   cuda::free(d_domainList);

}

void DomainListHandler::reset() {

    keyType maxKey = (keyType)KEY_MAX;
    cuda::set(d_domainListIndices, -1, domainListSize);
    cuda::set(d_domainListLevels, -1, domainListSize);
    cuda::set(d_domainListIndex, 0, 1);
    cuda::set(d_domainListCounter, 0, 1);
    cuda::set(d_domainListKeys, maxKey, domainListSize);
    cuda::set(d_sortedDomainListKeys, maxKey, domainListSize);
    cuda::set(d_relevantDomainListIndices, -1, domainListSize);
}

