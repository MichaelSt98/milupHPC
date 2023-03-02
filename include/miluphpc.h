/**
 * @file miluphpc.h
 * @brief Right-hand-side implementation and CUDA kernel execution via wrapper functions.
 *
 * **Abstract base class for integrator classes implementing the right-hand-side
 * via modular functions for different parts of the simulation.**
 *
 * Since dynamic memory allocation and access to heap objects in GPUs are usually suboptimal,
 * an array-based data structure is used to store the (pseudo-)particle information as well as the tree
 * and allows for efficient cache alignment. Consequently, array indices are used instead of pointers
 * to constitute the tree out of the tree nodes, whereas "-1" represents a null pointer and "-2" is
 * used for locking nodes. For this purpose an array with the minimum length \f$ 2^{d} \cdot (M - N) \f$
 * with dimensionality \f$ d \f$ and number of cells \f$ (M -N) \f$ is needed to store the children.
 * The \f$ i \f$-th child of node \f$ c \f$ can therefore be accessed via index \f$ 2^d \cdot c + i \f$.
 *
 * \image html images/Parallelization/coarse_flow.png width=50%
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
#include "mfv/volume_partition.cuh"
#include "mfv/riemann_fluxes.cuh"
#include "mfv/riemann_solver_handler.cuh"
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
 * @brief MilupHPC class
 *
 * **Abstract base class for integrator classes implementing the right-hand-side
 * via modular functions for different parts of the simulation.**
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
     * @brief Reset arrays, values, ...
     *
     * This function embraces resetting the
     *
     * * pseudo-particles
     * * child array
     * * the boundaries
     * * the domain list nodes
     * * the neighbor list
     * * and so forth ...
     *
     * ensuring correct starting conditions.
     *
     * @return accumulated time of functions within
     */
    real reset();

    /**
     * @brief Calculate bounding boxes/simulation domain.
     *
     * To derive the bounding boxes of the simulation domain the maximum and minimum value for each coordinate
     * axis is determined locally on each process and afterwards reduced to all processes to get the global
     * maximum and minimum via `MPI_Allreduce()`.
     *
     * @return accumulated time of functions within
     */
    real boundingBox();

    /**
     * @brief Parallel version regarding tree-stuff.
     *
     * @return accumulated time of functions within
     */
    real parallel_tree();

    /**
     * @brief Parallel version regarding computation of pseudo-particles.
     *
     * @return accumulated time of functions within
     */
    real parallel_pseudoParticles();

    /**
     * @brief Parallel version regarding computation of gravitational stuff.
     *
     * @return accumulated time of functions within
     */
    real parallel_gravity();

    /**
     * @brief Parallel version regarding computation of SPH-stuff.
     *
     * @return accumulated time of functions within
     */
    real parallel_sph();

    // @todo possible to combine sendPartclesEntry and sendParticles
    /**
     * @brief Send particles/Exchange particles among MPI processes.
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
     * @brief Send particles/Exchange particles among MPI processes.
     *
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
     * @brief Function to sort an array `entry` in dependence of another array `sortArray`
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
     * @brief Assign particles to correct process in dependence of particle key and ranges.
     *
     * First an extra array is used to save the information of process assignment for each particle,
     * this extra array is used as key for sorting all of the attributes of the Particle class
     * and finally the sub-array determined by the process assignment are sent to the corresponding process via MPI.
     *
     * @return accumulated time of functions within
     */
    real assignParticles();

    /**
     * @brief Calculate the angular momentum for all particles.
     *
     * @return accumulated time of functions within
     */
    real angularMomentum();

    /**
     * @brief Calculate the total amount of energy.
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
     * @brief Remove particles in dependence of some criterion.
     *
     * Criterion can be specified in the config file.
     * Removing particles can be accomplished by marking the particles to be removed via an extra array
     * and using this one as key for sorting all the (pseudo-)particle arrays and finally deleting them
     * by overwriting those entries with default values.
     *
     * @return accumulated time of functions within
     */
    real removeParticles();

    /**
     * @brief Load balancing via equidistant ranges.
     */
    void fixedLoadBalancing();

    /**
     * @brief Pre-calculations for `updateRangeApproximately`.
     *
     * Potential wrapper functions if more *range determination functions* come
     * into exist.
     *
     * @param bins amount of bins the range will be subdivided
     */
    void dynamicLoadBalancing(int bins=5000);

    /**
     * @brief Update the ranges (approximately and dynamically).
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
     * @brief Update the range in dependence on number of (MPI) processes and aimed particles per process.
     *
     * Counting the particles per process, calculating the aimed particles per process as quotient of total number of
     * particles and number of processes, determining the particle keys, sorting them and determining the new ranges
     * as the indices of the multiplies of the aimed number of particles per process.
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

    /// Instance to handle Riemann solvers on device and host
    MFV::RiemannSolverHandler riemannHandler;

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
    //HelperHandler *helperHandler;
    /// buffer instance
    HelperHandler *buffer;

    // testing
    /// buffer (need for revising)
    //integer *d_particles2SendIndices;
    /// buffer (need for revising)
    //integer *d_pseudoParticles2SendIndices;
    /// buffer (need for revising)
    //integer *d_pseudoParticles2SendLevels;
    /// buffer (need for revising)
    //integer *d_pseudoParticles2ReceiveLevels;
    /// buffer (need for revising)
    //integer *d_particles2SendCount;
    /// buffer (need for revising)
    //integer *d_pseudoParticles2SendCount;
    /// buffer (need for revising)
    //int *d_particles2removeBuffer;
    /// buffer (need for revising)
    //int *d_particles2removeVal;
    /// buffer (need for revising)
    //idInteger *d_idIntegerBuffer;
    /// buffer (need for revising)
    //idInteger *d_idIntegerCopyBuffer;
    // end: testing

    /// collected information required to set up the simulation
    SimulationParameters simulationParameters;

    /** Container for all parameters used for slope limiting (read in from the config file)
     *  on device ...
     */
    MFV::SlopeLimitingParameters *d_slopeLimitingParameters;
    /// ... and on host
    MFV::SlopeLimitingParameters h_slopeLimitingParameters;


    /**
     * @brief Constructor to set up simulation.
     *
     * @param simulationParameters all the information required to set up simulation
     */
    Miluphpc(SimulationParameters simulationParameters);

    /**
     * @brief Destructor freeing class instances.
     */
    ~Miluphpc();

    /**
     * @brief Prepare the simulation, including
     *
     * * loading the initial conditions
     * * copying to device
     * * computing the bounding boxes
     * * initial ranges (load balancing)
     */
    void prepareSimulation();

    /**
     * @brief Determine amount of particles (`numParticles` and `numParticlesLocal`)
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
     * @brief Read initial/particle distribution file (in parallel)
     *
     * @param filename initial conditions file to be read
     */
    void distributionFromFile(const std::string& filename);

    /**
     * @brief Wrapper function for building the tree (and domain tree).
     *
     * To build the parallel tree it is necessary to determine the domain list or common coarse tree nodes which
     * are derived from the ranges. After that, the tree is built locally and finally all the domain list nodes
     * or rather those attributes as indices are either saved to an instance of the omainList class
     * or if they do not exist after building the tree created and afterwards assigned.
     *
     * @return accumulated time for functions within
     */
    real tree();

    /**
     * @brief Wrapper function for calculating pseudo-particles
     *
     * The pseudo-particle computation starts with determining the lowest domain list, this information is stored
     * in a separate instance of the DomainList class continues with the local COM computation,
     * the resetting of the domain list nodes that are not lowest domain list nodes, the preparation for the
     * exchange by copying the information into contiguous memory, subsequent communication via MPI, the updating
     * of the lowest domain list nodes and finally ends with the computation of the remaining domain list node
     * pseudo-particles.
     *
     * @return accumulated time for functions within
     */
    real pseudoParticles();

    /**
     * @brief Wrapper function for Gravity-related stuff
     *
     * The gravitational force computation corresponds to the Barnes-Hut method. First all relevant lowest domain
     * list nodes belonging to another process are determined. Then the pseudo-particles and particles to be sent
     * can be determined by checking the extended $\theta$-criterion for all nodes and all relevant lowest domain
     * list nodes. This is done in an approach where particles are either marked to be sent or not in an extra array
     * and afterwards copied to contiguous memory in order to send them via MPI. After receiving, the particles are
     * inserted into the local tree, whereas first the pseudo-particles and then the particles are inserted.
     * Since complete sub-trees are inserted, no new pseudo-particles are generated during this process. Afterwards,
     * the actual force computation can be accomplished locally. Finally the inserted (pseudo-)particles are removed.
     *
     * \image html images/Parallelization/interactions_gravity.png width=30%
     *
     * @return accumulated time for functions within
     */
    real gravity();

    /**
     * Wrapper function for SPH-related stuff
     *
     * Similar to the gravitational force computation is the SPH part. The relevant lowest domain list nodes are
     * determined in order to decide which particles need to be sent, which is also done in an analogous approach
     * in the gravitational part.
     * However, in this case only particles are sent and after
     * exchanging those particles, the insertion includes generation of new pseudo-particles similar to building
     * or rather extending the local tree. With inserted particles from the other processes the FRNN search can be
     * done and subsequent computation of the density, speed of sound and pressure realized.
     * The corresponding particle properties are
     * sent again, so that the actual force computation can be performed. Finally, the inserted and generated
     * (pseudo-)particles are removed.
     *
     * \image html images/SPH_concept.png width=50%
     * \image html images/Parallelization/interactions_sph.png width=30%
     *
     * @return accumulated time for functions within
     */
    real sph();

    /**
     * Right Hand Side - all of the simulation without integration itself.
     *
     * This method is used/called by the different integrators in order to
     * integrate or execute the simulation.
     * The force or acceleration computation and necessary steps to calculate them are encapsulated in this function
     * whose individual components and interoperation are depicted in the following.
     * This right-hand-side is called once or several times within a (sub-)integration step in dependence of the
     * integration scheme. In addition, the function depends on whether the simulation runs single-node or multi-node,
     * gravity and/or SPH are included into the simulation and some other conditions.
     *
     * \image html images/Parallelization/rhs_flow.png width=50%
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

    void getMemoryInfo();

};

#endif //MILUPHPC_MILUPHPC_H
