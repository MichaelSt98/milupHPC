//
// Created by Michael Staneker on 15.08.21.
//

#ifndef MILUPHPC_MILUPHPC_H
#define MILUPHPC_MILUPHPC_H

#include "particle_handler.h"
#include "subdomain_key_tree/tree_handler.h"
#include "subdomain_key_tree/subdomain_handler.h"

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

public:

    integer numParticles;
    integer numParticlesLocal;
    integer numNodes;

    ParticleHandler *particleHandler;
    SubDomainKeyTreeHandler *subDomainKeyTreeHandler;

    Miluphpc(integer numParticles, integer numNodes);
    ~Miluphpc();

    void initDistribution(ParticleDistribution::Type particleDistribution=ParticleDistribution::disk);

};


#endif //MILUPHPC_MILUPHPC_H
