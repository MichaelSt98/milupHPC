//
// Created by Michael Staneker on 15.08.21.
//

#include "../../include/subdomain_key_tree/subdomain_handler.h"

SubDomainKeyTreeHandler::SubDomainKeyTreeHandler() {

    MPI_Comm_rank(MPI_COMM_WORLD, &h_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &h_numProcesses);
    h_range = new keyType[h_numProcesses + 1];

    h_subDomainKeyTree = new SubDomainKeyTree();
    h_subDomainKeyTree->set(h_rank, h_numProcesses, h_range);

    gpuErrorcheck(cudaMalloc((void**)&d_rank, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_numProcesses, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_range, (h_numProcesses + 1) * sizeof(keyType)));

    gpuErrorcheck(cudaMalloc((void**)&d_subDomainKeyTree, sizeof(SubDomainKeyTree)));
    SubDomainKeyTreeNS::launchSetKernel(d_subDomainKeyTree, d_rank, d_numProcesses, d_range);
}

SubDomainKeyTreeHandler::~SubDomainKeyTreeHandler() {

    delete [] h_range;
    delete h_subDomainKeyTree;
}


DomainListHandler::DomainListHandler(integer domainListSize) : domainListSize(domainListSize) {

    gpuErrorcheck(cudaMalloc((void**)&d_domainListIndices, domainListSize * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListLevels, domainListSize * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListIndex, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListCounter, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_domainListKeys, domainListSize * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_sortedDomainListKeys, domainListSize * sizeof(integer)));

    gpuErrorcheck(cudaMalloc((void**)&d_domainList, sizeof(DomainList)));
    DomainListNS::launchSetKernel(d_domainList, d_domainListIndices, d_domainListLevels,
                                  d_domainListIndex, d_domainListCounter, d_domainListKeys,
                                  d_sortedDomainListKeys);
}

DomainListHandler::~DomainListHandler() {

    gpuErrorcheck(cudaFree(d_domainListIndices));
    gpuErrorcheck(cudaFree(d_domainListLevels));
    gpuErrorcheck(cudaFree(d_domainListIndex));
    gpuErrorcheck(cudaFree(d_domainListCounter));
    gpuErrorcheck(cudaFree(d_domainListKeys));
    gpuErrorcheck(cudaFree(d_sortedDomainListKeys));
    gpuErrorcheck(cudaFree(d_domainList));

}

