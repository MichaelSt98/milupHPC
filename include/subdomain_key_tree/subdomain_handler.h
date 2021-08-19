//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_SUBDOMAIN_HANDLER_H
#define MILUPHPC_SUBDOMAIN_HANDLER_H

#include "../parameter.h"
#include "subdomain.cuh"
#include <mpi.h>

class KeyHandler {

};

class SubDomainKeyTreeHandler {

public:
    integer h_rank;
    integer h_numProcesses;
    keyType *h_range;

    integer *d_rank;
    integer *d_numProcesses;
    keyType *d_range;

    SubDomainKeyTree *h_subDomainKeyTree;
    SubDomainKeyTree *d_subDomainKeyTree;

    SubDomainKeyTreeHandler();
    ~SubDomainKeyTreeHandler();

};

class DomainListHandler {

public:
    integer domainListSize;

    integer *d_domainListIndices;
    integer *d_domainListLevels;
    integer *d_domainListIndex;
    integer *d_domainListCounter;
    keyType *d_domainListKeys;
    keyType *d_sortedDomainListKeys;

    DomainList *d_domainList;

    DomainListHandler(integer domainListSize);
    ~DomainListHandler();

};

#endif //MILUPHPC_SUBDOMAIN_HANDLER_H
