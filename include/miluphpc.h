#ifndef MILUPHPC_MILUPHPC_H
#define MILUPHPC_MILUPHPC_H

#include "particle_handler.h"
#include "subdomain_key_tree/tree_handler.h"
#include "subdomain_key_tree/subdomain_handler.h"
#include "device_rhs.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "utils/logger.h"
#include "utils/timer.h"
#include "materials/material_handler.h"
//#include "integrator/integrator.h"
#include "helper_handler.h"
#include "gravity/gravity.cuh"
#include "sph/sph.cuh"
#include "cuda_utils/cuda_runtime.h"
#include "utils/cxxopts.h"

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits> // for ulong_max
#include <algorithm>
#include <cmath>
#include <utility>
#include <set>
#include <fstream>
#include <iomanip>
#include <random>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5DataSet.hpp>

struct ParticleDistribution
{
    enum Type
    {
        disk, plummer
    };
    Type t_;
    ParticleDistribution(Type t) : t_(t) {}
    operator Type () const {return t_;}
private:
    template<typename T>
    operator T () const;
};

class Miluphpc {

private:
    void diskModel();
    void plummerModel();

    void updateRangeApproximately(int aimedParticlesPerProcess, int bins=4000);
    void fixedLoadDistribution();
    void dynamicLoadDistribution();

    //real reset();

    real compPseudoParticlesParallel();
    void parallelForce();

    //real serial_tree();
    //real serial_pseudoParticles();
    //real serial_gravity();
    //real serial_sph();

    real parallel_tree();
    real parallel_pseudoParticles();
    real parallel_gravity();
    real parallel_sph();

    real parallel_sph_backup();

    template <typename T>
    integer sendParticlesEntry(integer *sendLengths, integer *receiveLengths, T *entry);
    void exchangeParticleEntry(integer *sendLengths, integer *receiveLengths, real *entry);

    template <typename T>
    integer sendParticles(T *sendBuffer, T *receiveBuffer, integer *sendLengths, integer *receiveLengths);


    real reset();
    real boundingBox();

    template <typename T>
    real arrangeParticleEntries(T *entry, T *temp);

    real assignParticles();

public:

    Curve::Type curveType;

    integer numParticles;
    integer numParticlesLocal;
    integer numNodes;

    IntegratedParticles *integratedParticles;

    integer *d_mutex;
    HelperHandler *helperHandler; // TODO: more than one is needed: how to name?
    HelperHandler *buffer;
    ParticleHandler *particleHandler;
    SubDomainKeyTreeHandler *subDomainKeyTreeHandler;
    TreeHandler *treeHandler;
    DomainListHandler *domainListHandler;
    DomainListHandler *lowestDomainListHandler;

    // testing
    integer *d_particles2SendIndices;
    integer *d_pseudoParticles2SendIndices;
    integer *d_pseudoParticles2SendLevels;
    integer *d_pseudoParticles2ReceiveLevels;

    integer *d_particles2SendCount;
    integer *d_pseudoParticles2SendCount;
    // end: testing

    SimulationParameters simulationParameters;

    Miluphpc(SimulationParameters simulationParameters, integer numParticles, integer numNodes);
    ~Miluphpc();

    /**
     *
     * @param particleDistribution
     */
    void initDistribution(ParticleDistribution::Type particleDistribution=ParticleDistribution::disk);
    void loadDistribution();

    void distributionFromFile();

    real tree();
    real pseudoParticles();
    real gravity();
    real sph();

    real rhs(int step);

    //virtual void integrate() {};
    virtual void integrate(int step = 0) = 0;

    float particles2file(HighFive::DataSet *pos, HighFive::DataSet *vel, HighFive::DataSet *key);

};


#endif //MILUPHPC_MILUPHPC_H
