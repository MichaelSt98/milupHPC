//
// Created by Michael Staneker on 15.08.21.
//

#include "../../include/subdomain_key_tree/subdomain_handler.h"

SubDomainKeyTreeHandler::SubDomainKeyTreeHandler() {

    boost::mpi::communicator comm;

    h_rank = comm.rank();
    h_numProcesses = comm.size();
    h_range = new keyType[h_numProcesses + 1];
    h_procParticleCounter = new integer[h_numProcesses];

    h_subDomainKeyTree = new SubDomainKeyTree();
    h_subDomainKeyTree->set(h_rank, h_numProcesses, h_range, h_procParticleCounter);

    gpuErrorcheck(cudaMalloc((void**)&d_rank, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_numProcesses, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_range, (h_numProcesses + 1) * sizeof(keyType)));
    gpuErrorcheck(cudaMalloc((void**)&d_procParticleCounter, h_numProcesses * sizeof(integer)));

    gpuErrorcheck(cudaMalloc((void**)&d_subDomainKeyTree, sizeof(SubDomainKeyTree)));
    SubDomainKeyTreeNS::Kernel::Launch::set(d_subDomainKeyTree, h_rank, h_numProcesses, d_range, d_procParticleCounter);
    //gpuErrorcheck(cudaMemcpy(d_rank, &h_rank, sizeof(integer), cudaMemcpyHostToDevice));
    //gpuErrorcheck(cudaMemcpy(d_numProcesses, &h_numProcesses, sizeof(integer), cudaMemcpyHostToDevice));
}

SubDomainKeyTreeHandler::~SubDomainKeyTreeHandler() {

    delete [] h_range;
    delete [] h_procParticleCounter;
    delete h_subDomainKeyTree;

    gpuErrorcheck(cudaFree(d_range));
    gpuErrorcheck(cudaFree(d_procParticleCounter));
    gpuErrorcheck(cudaFree(d_subDomainKeyTree));
}

void SubDomainKeyTreeHandler::toDevice() {
    gpuErrorcheck(cudaMemcpy(d_range, h_range, (h_numProcesses + 1) * sizeof(keyType), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_procParticleCounter, h_procParticleCounter,
                             h_numProcesses* sizeof(integer), cudaMemcpyHostToDevice));
}

void SubDomainKeyTreeHandler::toHost() {
    gpuErrorcheck(cudaMemcpy(h_range, d_range, (h_numProcesses + 1) * sizeof(keyType), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(h_procParticleCounter, d_procParticleCounter, h_numProcesses * sizeof(integer),
                             cudaMemcpyDeviceToHost));
}

void SubDomainKeyTreeHandler::reset() {
    gpuErrorcheck(cudaMemset(d_procParticleCounter, 0, h_numProcesses  * sizeof(integer)));
    for (int i=0; i<h_numProcesses; i++) {
        h_procParticleCounter[i] = 0;
    }
}

DomainListHandler::DomainListHandler(integer domainListSize) : domainListSize(domainListSize) {

    gpuErrorcheck(cudaMalloc((void**)&d_domainListIndices, domainListSize * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListLevels, domainListSize * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListIndex, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListCounter, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListKeys, domainListSize * sizeof(keyType)));
    gpuErrorcheck(cudaMalloc((void**)&d_sortedDomainListKeys, domainListSize * sizeof(keyType)));
    gpuErrorcheck(cudaMalloc((void**)&d_relevantDomainListIndices, domainListSize * sizeof(integer)));

    gpuErrorcheck(cudaMalloc((void**)&d_domainList, sizeof(DomainList)));
    DomainListNS::Kernel::Launch::set(d_domainList, d_domainListIndices, d_domainListLevels,
                                      d_domainListIndex, d_domainListCounter, d_domainListKeys,
                                      d_sortedDomainListKeys, d_relevantDomainListIndices);
}

DomainListHandler::~DomainListHandler() {

    gpuErrorcheck(cudaFree(d_domainListIndices));
    gpuErrorcheck(cudaFree(d_domainListLevels));
    gpuErrorcheck(cudaFree(d_domainListIndex));
    gpuErrorcheck(cudaFree(d_domainListCounter));
    gpuErrorcheck(cudaFree(d_domainListKeys));
    gpuErrorcheck(cudaFree(d_sortedDomainListKeys));
    gpuErrorcheck(cudaFree(d_relevantDomainListIndices));
    gpuErrorcheck(cudaFree(d_domainList));

}

void DomainListHandler::reset() {

    keyType maxKey = (keyType)KEY_MAX;
    gpuErrorcheck(cudaMemset(d_domainListIndices, -1, domainListSize * sizeof(integer)));
    gpuErrorcheck(cudaMemset(d_domainListLevels, -1, domainListSize * sizeof(integer)));
    gpuErrorcheck(cudaMemset(d_domainListIndex, 0, sizeof(integer)));
    gpuErrorcheck(cudaMemset(d_domainListCounter, 0, sizeof(integer)));
    gpuErrorcheck(cudaMemset(d_domainListKeys, maxKey, domainListSize * sizeof(keyType)));
    gpuErrorcheck(cudaMemset(d_sortedDomainListKeys, maxKey, domainListSize * sizeof(keyType)));
    gpuErrorcheck(cudaMemset(d_relevantDomainListIndices, -1, domainListSize * sizeof(integer)));
}

