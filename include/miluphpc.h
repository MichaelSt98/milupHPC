//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_MILUPHPC_H
#define MILUPHPC_MILUPHPC_H

#include "particle_handler.h"
#include "subdomain_key_tree/tree_handler.h"
#include "subdomain_key_tree/subdomain_handler.h"
#include "device_rhs.cuh"
#include "cuda_utils/cuda_utilities.cuh"
#include "utils/logger.h"
#include "materials/material_handler.h"
#include "integrator/integrator.h"
#include "helper_handler.h"
#include "gravity/gravity.cuh"

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
    void diskModel(Curve::Type curveType=Curve::lebesgue);

    void updateRangeApproximately(int aimedParticlesPerProcess, int bins=4000);
    void fixedLoadDistribution();
    void dynamicLoadDistribution();

    void compPseudoParticlesParallel();
    void parallelForce();

    template <typename T>
    integer sendParticlesEntry(integer *sendLengths, integer *receiveLengths, T *entry);
    void exchangeParticleEntry(integer *sendLengths, integer *receiveLengths, real *entry);

public:

    integer numParticles;
    integer numParticlesLocal;
    integer numNodes;

    integer *d_mutex;
    HelperHandler *helperHandler; // TODO: more than one is needed: how to name?
    HelperHandler *buffer;
    ParticleHandler *particleHandler;
    SubDomainKeyTreeHandler *subDomainKeyTreeHandler;
    TreeHandler *treeHandler;
    DomainListHandler *domainListHandler;
    DomainListHandler *lowestDomainListHandler;

    Miluphpc(integer numParticles, integer numNodes);
    ~Miluphpc();

    void initDistribution(ParticleDistribution::Type particleDistribution=ParticleDistribution::disk,
                          Curve::Type curveType=Curve::lebesgue);
    void initBarnesHut();

    void barnesHut();
    void run();


    void particles2file(HighFive::DataSet *pos, HighFive::DataSet *vel, HighFive::DataSet *key);

};


#endif //MILUPHPC_MILUPHPC_H
