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

    void updateRangeApproximately(int aimedParticlesPerProcess, int bins=4000);
    void newLoadDistribution();

public:

    integer numParticles;
    integer numParticlesLocal;
    integer numNodes;

    integer *d_mutex;
    HelperHandler *helperHandler;
    HelperHandler *buffer;
    ParticleHandler *particleHandler;
    SubDomainKeyTreeHandler *subDomainKeyTreeHandler;
    TreeHandler *treeHandler;

    Miluphpc(integer numParticles, integer numNodes);
    ~Miluphpc();

    void initDistribution(ParticleDistribution::Type particleDistribution=ParticleDistribution::disk);
    void barnesHut();
    void run();

};


#endif //MILUPHPC_MILUPHPC_H
