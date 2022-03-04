/**
 * @file miluphpc.h
 * @brief Right-hand-side implementation and CUDA kernel execution via wrapper functions.
 *
 * Abstract bass class for integrator classes implementing the right-hand-side
 * via modular functions for different parts of the simulation.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
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
#include "sph/soundspeed.cuh"
#include "simulation_time_handler.h"

#include "processing/kernels.cuh"

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

#include <boost/mpi/collectives/all_gatherv.hpp>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5DataSet.hpp>

/**
 * Miluphpc class
 *
 * More detailed description ...
 */
class Miluphpc {

private:

    // @todo implement serial versions in order to reduce overhead
    //  arised from parallelization
    //real serial_tree();
    //real serial_pseudoParticles();
    //real serial_gravity();
    //real serial_sph();

    /**
     * Reset arrays, values, ...
     *
     * @return accumulated time of functions within
     */
    real reset();

    /**
     * Calculate bounding boxes/simulation domain
     * @return accumulated time of functions within
     */
    real boundingBox();

    /**
     * Parallel version regarding tree-stuff
     * @return accumulated time of functions within
     */
    real parallel_tree();

    /**
     * Parallel version regarding computation of pseudo-particles.
     * @return accumulated time of functions within
     */
    real parallel_pseudoParticles();

    /**
     * Parallel version regarding computation of gravitational stuff.
     * @return accumulated time of functions within
     */
    real parallel_gravity();

    /**
     * Parallel version regarding computation of SPH-stuff.
     * @return accumulated time of functions within
     */
    real parallel_sph();

    // @todo possible to combine sendPartclesEntry and sendParticles
    /**
     * Send particles/Exchange particles among MPI processes.
     *
     * @tparam T
     * @param sendLengths
     * @param receiveLengths
     * @param entry
     * @param entryBuffer
     * @param copyBuffer
     * @return
     */
    template <typename T>
    integer sendParticlesEntry(integer *sendLengths, integer *receiveLengths, T *entry, T *entryBuffer, T *copyBuffer);
    //void exchangeParticleEntry(integer *sendLengths, integer *receiveLengths, real *entry);

    /**
     * Send particles/Exchange particles among MPI processes.
     * @tparam T
     * @param sendBuffer
     * @param receiveBuffer
     * @param sendLengths
     * @param receiveLengths
     * @return
     */
    template <typename T>
    integer sendParticles(T *sendBuffer, T *receiveBuffer, integer *sendLengths, integer *receiveLengths);

    /**
     * Function to sort an array `entry` in dependence of another array `sortArray`
     *
     * @tparam U type of the arrays determining the sorting
     * @tparam T type of the arrays to be sorted
     * @param sortArray array to be sorted (rather sorting behaviour should be determined)
     * @param sortedArray sorted array (result of sorting)
     * @param entry (relevant) array to be sorted
     * @param temp buffer needed for sorting process
     * @return accumulated time of functions within
     */
    template <typename U, typename T>
    real arrangeParticleEntries(U *sortArray, U *sortedArray, T *entry, T *temp);

    /**
     * Assign particles to correct process in dependence of particle key and ranges.
     *
     * @return accumulated time of functions within
     */
    real assignParticles();

    /**
     * Calculate the angular momentum for all particles.
     *
     * @return accumulated time of functions within
     */
    real angularMomentum();

    /**
     * Calculate the total amount of energy.
     *
     * @return accumulated time of functions within
     */
    real energy();

public:

    /// current sub-step (there are possibly more sub-steps within a step!)
    int subStep;

    /// search radius for SPH (MPI-process overarching) neighbor search
    real h_searchRadius;

    /// total energy
    real totalEnergy;

    /// total angular momentum
    real totalAngularMomentum;

    /**
     * Remove particles in dependence of some criterion.
     *
     * Criterion can be specified in the config file.
     *
     * @return accumulated time of functions within
     */
    real removeParticles();

    /**
     * Load balancing via equidistant ranges.
     */
    void fixedLoadBalancing();

    /**
     * Pre-calculations for `updateRangeApproximately`.
     *
     * Potential wrapper functions if more *range determination functions* come
     * into exist.
     *
     * @param bins amount of bins the range will be subdivided
     */
    void dynamicLoadBalancing(int bins=5000);

    /**
     * Update the ranges (approximately and dynamically).
     *
     * A histogram is generated with the amount of bins given by `bins`.
     * The possible range is distributed accordingly and particles are sorted
     * in dependence of their key into, in order to extract the ranges.
     *
     * @param aimedParticlesPerProcess aimed amount particles per process
     * @param bins amount of bins the range will be subdivided
     */
    void updateRangeApproximately(int aimedParticlesPerProcess, int bins=5000);

    /**
     * Update the range in dependence on number of (MPI) processes and aimed particles per process.
     *
     * @param aimedParticlesPerProcess optimal number of particles per process
     */
    void updateRange(int aimedParticlesPerProcess);

    /// H5 profiler instance
    H5Profiler &profiler = H5Profiler::getInstance("log/performance.h5");

    /// Space-filling curve type to be used (Lebesgue or Hilbert)
    Curve::Type curveType;

    /// Instance to handle the SPH `Kernel` instance on device and host
    SPH::KernelHandler kernelHandler;
    /// Instance to handle the `SimulationTime` instances on device and host
    SimulationTimeHandler *simulationTimeHandler;

    /// number of particles (to be allocated)
    integer numParticles;
    /// (real) number of particles on all processes
    integer sumParticles;
    /**
     * number of particles currently living on the (MPI) process
     *
     * Temporarily are more particles on the process, e.g. for gravitational
     * and SPH (force) calculations.
     */
    integer numParticlesLocal;
    /// number of nodes (to be allocated)
    integer numNodes;

    /**
     * Instance(s) to handle the `IntegratedParticles` instance(s) on device and host
     *
     * Memory for this may or may not be allocated within child classes.
     * This depends on whether instances are needed for cache particle information
     * e.g. for predictor-corrector integrators.
     */
    IntegratedParticleHandler *integratedParticles;

    integer *d_mutex;
    /// Instance to handle the `Particles` instance on device and host
    ParticleHandler *particleHandler;
    /// Instance to handle the `SubDomainKeyTree` instance on device and host
    SubDomainKeyTreeHandler *subDomainKeyTreeHandler;
    /// Instance to handle the `Tree` instance on device and host
    TreeHandler *treeHandler;
    /// Instance to handle the `DomainList` instance on device and host
    DomainListHandler *domainListHandler;
    /// Instance to handle the (lowest) `DomainList` instance on device and host
    DomainListHandler *lowestDomainListHandler;
    /// Instance to handle `Materials` instances on device and host
    MaterialHandler *materialHandler;

    /// @todo revise buffer handling
    /// buffer instance
    HelperHandler *helperHandler;
    /// buffer instance
    HelperHandler *buffer;

    // testing
    /// buffer (need for revising)
    integer *d_particles2SendIndices;
    /// buffer (need for revising)
    integer *d_pseudoParticles2SendIndices;
    /// buffer (need for revising)
    integer *d_pseudoParticles2SendLevels;
    /// buffer (need for revising)
    integer *d_pseudoParticles2ReceiveLevels;
    /// buffer (need for revising)
    integer *d_particles2SendCount;
    /// buffer (need for revising)
    integer *d_pseudoParticles2SendCount;
    /// buffer (need for revising)
    int *d_particles2removeBuffer;
    /// buffer (need for revising)
    int *d_particles2removeVal;
    /// buffer (need for revising)
    idInteger *d_idIntegerBuffer;
    /// buffer (need for revising)
    idInteger *d_idIntegerCopyBuffer;
    // end: testing

    /// collected information required to set up the simulation
    SimulationParameters simulationParameters;

    /**
     * Constructor to set up simulation.
     *
     * The distance between \f$(x_1,y_1)\f$ and \f$(x_2,y_2)\f$ is \f$\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}\f$.
     *
     * \f{equation}{ x=2 \f}
     *
     * @param simulationParameters all the information required to set up simulation
     */
    Miluphpc(SimulationParameters simulationParameters);

    /**
     * Destructor freeing class instances.
     */
    ~Miluphpc();

    /**
     * Prepare the simulation, including
     *
     * * loading the initial conditions
     * * copying to device
     * * computing the bounding boxes
     * * initial ranges (load balancing)
     */
    void prepareSimulation();

    /**
     * Determine amount of particles (`numParticles` and `numParticlesLocal`)
     * from initial file/particle distribution file
     *
     * Since the information of how many particles to be simulated is needed to
     * allocate (the right amount of) memory, this function need to be called before
     * `distributionFromFile()`
     *
     * @param filename initial conditions file to be read
     */
    void numParticlesFromFile(const std::string& filename);

    /**
     * Read initial/particle distribution file (in parallel)
     *
     * @param filename initial conditions file to be read
     */
    void distributionFromFile(const std::string& filename);

    /**
     * Wrapper function for building the tree (and domain tree)
     * @return accumulated time for functions within
     */
    real tree();

    /**
     * Wrapper function for calculating pseudo-particles
     * @return accumulated time for functions within
     */
    real pseudoParticles();

    /**
     * Wrapper function for Gravity-related stuff
     * @return accumulated time for functions within
     */
    real gravity();

    /**
     * Wrapper function for SPH-related stuff
     * @return accumulated time for functions within
     */
    real sph();

    /**
     * Right Hand Side - all of the simulation without integration itself.
     *
     * This method is used/called by the different integrators in order to
     * integrate or execute the simulation.
     *
     * @param step simulation step, regarding output
     * @param selfGravity apply self-gravity y/n (needed since gravity is decoupled)
     * @param assignParticlesToProcess assign particles to correct process y/n
     * @return accumulated time for functions within
     */
    real rhs(int step, bool selfGravity=true, bool assignParticlesToProcess=true);

    /**
     * Abstract method for integration. Implemented by integration classes,
     * calling `rhs()` different times
     * @param step
     */
    virtual void integrate(int step = 0) = 0;

    /**
     * Function to be called after successful integration (step)
     */
    void afterIntegrationStep();

    /**
     * Write all of the information/particle distribution to H5 file
     *
     * @param step simulation step, used to name file
     * @return accumulated time for functions within
     */
    real particles2file(int step);

    /**
     * Write particles or rather particle's information/attributes to file. Particles to be outputted
     * determined by `particleIndices`
     *
     * @param filename file name of output H5 file
     * @param particleIndices particle's indices to be outputted
     * @param length amount of particles to be outputted
     * @return accumulated time for functions within
     */
    real particles2file(const std::string& filename, int *particleIndices, int length);

};

#endif //MILUPHPC_MILUPHPC_H
