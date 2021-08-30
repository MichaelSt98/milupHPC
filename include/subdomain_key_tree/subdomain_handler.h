//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_SUBDOMAIN_HANDLER_H
#define MILUPHPC_SUBDOMAIN_HANDLER_H

#include "../parameter.h"
#include "subdomain.cuh"
//#include <mpi.h>
#include <boost/mpi.hpp>

class KeyHandler {

};

class SubDomainKeyTreeHandler {

public:
    integer h_rank;
    integer h_numProcesses;
    keyType *h_range;
    integer *h_procParticleCounter;

    integer *d_rank;
    integer *d_numProcesses;
    keyType *d_range;
    integer *d_procParticleCounter;

    SubDomainKeyTree *h_subDomainKeyTree;
    SubDomainKeyTree *d_subDomainKeyTree;

    SubDomainKeyTreeHandler();
    ~SubDomainKeyTreeHandler();

    void toDevice();
    void toHost();

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
    integer *d_relevantDomainListIndices;

    DomainList *d_domainList;

    DomainListHandler(integer domainListSize);
    ~DomainListHandler();

};

namespace mpi {

    template <typename T>
    void messageLengths(SubDomainKeyTree *subDomainKeyTree, T *toSend, T *toReceive) {

        //boost::mpi::environment env;
        boost::mpi::communicator comm;

        std::vector <boost::mpi::request> reqParticles;
        std::vector <boost::mpi::status> statParticles;

        for (int proc = 0; proc < subDomainKeyTree->numProcesses; proc++) {
            if (proc != subDomainKeyTree->rank) {
                reqParticles.push_back(comm.isend(proc, 17, &toSend[proc], 1));
                statParticles.push_back(comm.recv(proc, 17, &toReceive[proc], 1));
            }
        }

        boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    }

}

#endif //MILUPHPC_SUBDOMAIN_HANDLER_H
