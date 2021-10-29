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
#include "helper_handler.h"
#include "gravity/gravity.cuh"
#include "sph/sph.cuh"
#include "cuda_utils/cuda_runtime.h"
#include "utils/cxxopts.h"
#include "utils/h5profiler.h"
#include "sph/kernel.cuh"
#include "sph/kernel_handler.cuh"
#include "sph/density.cuh"
#include "sph/pressure.cuh"
#include "sph/internal_forces.cuh"

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
#include <fstream>

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

    real removeParticles();

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

    //real parallel_sph_backup();

    template <typename T>
    integer sendParticlesEntry(integer *sendLengths, integer *receiveLengths, T *entry, T *entryBuffer, T *copyBuffer);
    //void exchangeParticleEntry(integer *sendLengths, integer *receiveLengths, real *entry);

    template <typename T>
    integer sendParticles(T *sendBuffer, T *receiveBuffer, integer *sendLengths, integer *receiveLengths);


    real reset();
    real boundingBox();

    template <typename T>
    real arrangeParticleEntries(T *entry, T *temp);

    template <typename U, typename T>
    real arrangeParticleEntries(U *sortArray, U *sortedArray, T *entry, T *temp);

    real assignParticles();


    real angularMomentum();
    real energy();

public:

    void fixedLoadBalancing();
    void dynamicLoadBalancing(int bins=5000);
    void updateRangeApproximately(int aimedParticlesPerProcess, int bins=5000);

    H5Profiler &profiler = H5Profiler::getInstance("log/performance.h5");
    Curve::Type curveType;

    SPH::KernelHandler kernelHandler;

    integer numParticles;
    integer sumParticles;
    integer numParticlesLocal;
    integer numNodes;

    IntegratedParticleHandler *integratedParticles;

    integer *d_mutex;
    HelperHandler *helperHandler; // TODO: more than one is needed: how to name?
    HelperHandler *buffer;
    ParticleHandler *particleHandler;
    SubDomainKeyTreeHandler *subDomainKeyTreeHandler;
    TreeHandler *treeHandler;
    DomainListHandler *domainListHandler;
    DomainListHandler *lowestDomainListHandler;

    MaterialHandler *materialHandler;

    // testing
    integer *d_particles2SendIndices;
    integer *d_pseudoParticles2SendIndices;
    integer *d_pseudoParticles2SendLevels;
    integer *d_pseudoParticles2ReceiveLevels;

    integer *d_particles2SendCount;
    integer *d_pseudoParticles2SendCount;

    int *d_particles2removeBuffer;
    int *d_particles2removeVal;

    idInteger *d_idIntegerBuffer;
    idInteger *d_idIntegerCopyBuffer;
    // end: testing

    SimulationParameters simulationParameters;

    //Miluphpc(SimulationParameters simulationParameters, integer numParticles, integer numNodes);
    //Miluphpc(SimulationParameters simulationParameters, const std::string& filename);
    Miluphpc(SimulationParameters simulationParameters);

    ~Miluphpc();

    /**
     *
     * @param particleDistribution
     */
    //void initDistribution(ParticleDistribution::Type particleDistribution=ParticleDistribution::disk);
    void prepareSimulation();

    void numParticlesFromFile(const std::string& filename);
    void distributionFromFile(const std::string& filename);

    real tree();
    real pseudoParticles();
    real gravity();
    real sph();

    real rhs(int step, bool selfGravity=true);

    //virtual void integrate() {};
    virtual void integrate(int step = 0) = 0;

    real particles2file(int step);

    real particles2file(const std::string& filename, int *particleIndices, int length);

};


#endif //MILUPHPC_MILUPHPC_H
