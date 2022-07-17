/**
 * @file subdomain_handler.h
 *
 * @brief Classes and Kernels handling subdomains that distribute the
 * workload among the MPI processes.
 *
 * @author Michael Staneker
 *
 */
#ifndef MILUPHPC_SUBDOMAIN_HANDLER_H
#define MILUPHPC_SUBDOMAIN_HANDLER_H

#include "../parameter.h"
#include "subdomain.cuh"
#include <boost/mpi.hpp>
#include <algorithm>

//class KeyHandler {
//};

/**
 * Handler for class `SubDomainKeyTree`
 */
class SubDomainKeyTreeHandler {

public:
    /// host MPI rank
    integer h_rank;
    /// host MPI number of processes
    integer h_numProcesses;
    /// host range(s)
    keyType *h_range;
    /// host counter for particles in dependence of MPI process belonging
    integer *h_procParticleCounter;

#if TARGET_GPU
    /// device MPI rank
    integer *d_rank;
    /// device MPI number of processes
    integer *d_numProcesses;
    /// device range(s)
    keyType *d_range;
    //keyType *d_hilberRange;
    /// host counter for particles in dependence of MPI process belonging
    integer *d_procParticleCounter;
#endif // TARGET_GPU

    /// host instance of class `SubDomainKeyTree`
    SubDomainKeyTree *h_subDomainKeyTree;
#if TARGET_GPU
    /// device instance of class `SubDomainKeyTree`
    SubDomainKeyTree *d_subDomainKeyTree;
#endif // TARGET_GPU

    /**
     * @brief Constructor
     */
    SubDomainKeyTreeHandler();

    /**
     * @brief Destructor
     */
    ~SubDomainKeyTreeHandler();

    /**
     * @brief Resetting member variables
     *
     * Setting counter for particles in dependence of MPI process to zero
     */
    void reset();

    /**
     * @brief Copy (parts of the) SubDomainKeyTree instance(s) between host and device
     *
     * @param target copy to target
     * @param range flag whether range(s) should be copied
     * @param counter flag whether counter should be copied
     */
    void copy(To::Target target=To::device, bool range=true, bool counter=true);

};

/**
 * Handler for class `DomainList`
 */
class DomainListHandler {

public:
    /// length/size of domain list
    integer domainListSize;

    // TODO: CPU necessary to have domain list ?

    /// host domain list indices
    integer *h_domainListIndices;
    /// host domain list levels
    integer *h_domainListLevels;
    /// host domain list index
    integer *h_domainListIndex;
    /// host domain list counter
    integer *h_domainListCounter;
    /// host domain list key
    keyType *h_domainListKeys;
    /// host sorted domain list keys
    keyType *h_sortedDomainListKeys;
    /// host relevant domain list indices
    integer *h_relevantDomainListIndices;
    integer *h_relevantDomainListLevels;
    integer *h_relevantDomainListProcess;
    integer *h_relevantDomainListOriginalIndex;
    real *h_borders;

    /// device instance of `DomainList` class
    DomainList *h_domainList;

#if TARGET_GPU
    /// device domain list indices
    integer *d_domainListIndices;
    /// device domain list levels
    integer *d_domainListLevels;
    /// device domain list index
    integer *d_domainListIndex;
    /// device domain list counter
    integer *d_domainListCounter;
    /// device domain list key
    keyType *d_domainListKeys;
    /// device sorted domain list keys
    keyType *d_sortedDomainListKeys;
    /// device relevant domain list indices
    integer *d_relevantDomainListIndices;
    integer *d_relevantDomainListLevels;
    integer *d_relevantDomainListProcess;
    integer *d_relevantDomainListOriginalIndex;
    real *d_borders;

    /// device instance of `DomainList` class
    DomainList *d_domainList;
#endif
    /**
     * @brief Constructor
     *
     * @param domainListSize size/length of domain lists
     */
    DomainListHandler(integer domainListSize);

    /**
     * @brief Destructor
     */
    ~DomainListHandler();

    /**
     * @brief Resetting entries
     */
    void reset();

};

namespace mpi {
    /**
     * @brief Send array with length of number of MPI processes across processes
     *
     * @tparam T type
     * @param subDomainKeyTree instance of `SubDomainKeyTree` class
     * @param toSend entries/array to be send
     * @param toReceive buffer/array for receiving
     */
    template <typename T>
    void messageLengths(SubDomainKeyTree *subDomainKeyTree, T *toSend, T *toReceive) {

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
