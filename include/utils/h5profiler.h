/**
 * @file h5profiler.h
 * @brief profiler based on (parallel) HDF5
 *
 * > Acknowledgment: Johannes Martin (GitHub: jammartin)
 * changed to be resizable via HighFive::Chunking
 *
 * @author Johannes Martin, Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_H5PROFILER_H
#define MILUPHPC_H5PROFILER_H


#include <string>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

#include "logger.h"

/// profiler identifiers
namespace ProfilerIds {

    const char* const numParticles      { "/general/numParticles" };
    const char* const numParticlesLocal { "/general/numParticlesLocal" };
    const char* const ranges            { "/general/ranges" };

    /// send lengths
    namespace SendLengths {
        const char* const gravityParticles         { "/sending/gravityParticles" };
        const char* const gravityPseudoParticles   { "/sending/gravityPseudoParticles" };
        const char* const sph             { "/sending/sph" };
    };

    /// receive lengths
    namespace ReceiveLengths {
        const char* const gravityParticles         { "/receiving/gravityParticles" };
        const char* const gravityPseudoParticles   { "/receiving/gravityPseudoParticles" };
        const char* const sph             { "/receiving/sph" };
    };

    /// timing/performance
    namespace Time {

        const char *const rhs             { "/time/rhs" };
        const char *const rhsElapsed      { "/time/rhsElapsed" };
        const char *const loadBalancing   { "/time/loadBalancing" };
        const char *const reset           { "/time/reset" };
        const char *const removeParticles { "/time/removeParticles" };
        const char *const boundingBox     { "/time/boundingBox" };
        const char *const assignParticles { "/time/assignParticles" };
        const char *const tree            { "/time/tree" };
        const char *const pseudoParticle  { "/time/pseudoParticle"};
        const char *const gravity         { "/time/gravity" };
        const char *const sph             { "/time/sph" };
        const char *const integrate       { "/time/integrate" };
        const char *const IO              { "/time/IO" };

        namespace Reset {

        }

        namespace BoundingBox {

        }

        namespace AssignParticles {

        }

        namespace Tree {
            const char *const createDomain    { "/time/tree_createDomainList" };
            const char *const tree            { "/time/tree_buildTree" };
            const char *const buildDomain     { "/time/tree_buildDomainTree" };
        }

        namespace PseudoParticles {

        }

        namespace Gravity {
            const char *const compTheta      { "/time/gravity_compTheta" };
            const char *const symbolicForce  { "/time/gravity_symbolicForce" };
            const char *const sending        { "/time/gravity_gravitySendingParticles" };
            const char *const insertReceivedPseudoParticles { "/time/gravity_insertReceivedPseudoParticles" };
            const char *const insertReceivedParticles       { "/time/gravity_insertReceivedParticles" };
            const char *const force          { "/time/gravity_force" };
            const char *const repairTree     { "/time/gravity_repairTree" };
        }

        namespace SPH {
            const char *const compTheta      { "/time/sph_compTheta" };
            const char *const determineSearchRadii { "time/sph_determineSearchRadii" };
            const char *const symbolicForce  { "/time/sph_symbolicForce" };
            const char *const sending        { "/time/sph_sendingParticles" };
            const char *const insertReceivedParticles { "/time/sph_insertReceivedParticles" };
            const char *const fixedRadiusNN  { "/time/sph_fixedRadiusNN" };
            const char *const density        { "/time/sph_density" };
            const char *const soundSpeed     { "/time/sph_soundSpeed" };
            const char *const pressure       { "/time/sph_pressure" };
            const char *const resend         { "/time/sph_resendingParticles" };
            const char *const internalForces { "/time/sph_internalForces" };
            const char *const repairTree     { "/time/sph_repairTree" };
        }

        namespace MFV {
            const char *const density { "/time/mfv_density" };
            const char *const vectorWeights { "/time/mfv_vectorWeights" };
            const char *const riemannFluxes { "/time/mfv_riemannFluxes" };
        }
    }

}

/**
 * @brief Singleton class for HDF5 profiler.
 */
class H5Profiler {
public:

    /**
     * @brief Constructor/Instance getter for HDF5 profiler.
     *
     * constructor is only called ONCE due to the singleton pattern,
     * arguments have to be passed when first calling getInstance()
     *
     * @param[in] outfile file to write
     * @return instance of HDF5 profiler
     */
    static H5Profiler& getInstance(const std::string& outfile = ""){
        static H5Profiler instance_(outfile);
        return instance_;
    }

    // deleting methods we don't want for the singleton design pattern (c++11)
    H5Profiler(H5Profiler const&) = delete;
    void operator=(H5Profiler const&) = delete;

    /**
     * @brief Set current step of profiler.
     *
     * @param[in] _step simulation step
     */
    void const setStep(const int& _step) { step = _step; }
    /**
     * @brief Set MPI rank.
     *
     * @param _myRank MPI rank
     */
    void const setRank(const int& _myRank) { myRank = _myRank; }
    /**
     * @brief Set number of MPI processes.
     *
     * @param _numProcs number of MPI processes
     */
    void const setNumProcs(const int& _numProcs) { numProcs = _numProcs; }
    /**
     * @brief Disable write to output file.
     */
    void const disableWrite() { disabled = true; }
    /**
     * @brief Enable write to output file.
     */
    void const enableWrite() { disabled = false; }

    int currentMaxLength = 0;

    /*void createTimeDataSet(const std::string& path, int steps);
    void time(const std::string &path);
    void timePause(const std::string &path);
    void time2file(const std::string &path, int myRank, bool onlyWrite=false);*/

    // TEMPLATE FUNCTIONS

    // track single values
    /**
     * @brief Track single value (per MPI rank).
     *
     * @tparam T data set data type
     * @param path HDF5 *path* to data/key
     * @param steps
     * @param maxSteps
     */
    template<typename T>
    void createValueDataSet(const std::string& path, int steps, std::size_t maxSteps=HighFive::DataSpace::UNLIMITED){
        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking(std::vector<hsize_t>{1, 1}));
        dataSets[path] = h5file.createDataSet<T>(path, HighFive::DataSpace({std::size_t(steps), std::size_t(numProcs)},
                                                                        {maxSteps, std::size_t(numProcs)}), props);
        totalSteps[path] = steps;
        currentSteps[path] = 0;
    }

    /**
     * @brief Write value to single value data set.
     *
     * @tparam T data set data type
     * @param path HDF5 *path* to data/key
     * @param value value to be written
     */
    template<typename T>
    void value2file(const std::string& path, T value) {

        if (!disabled) {
            if (currentSteps[path] >= totalSteps[path]) {
                totalSteps[path] += 1;
                //Logger(INFO) << "resizing: " << totalSteps[path];
                dataSets[path].resize({std::size_t(totalSteps[path]), std::size_t(numProcs)});
            }
            dataSets[path].select({std::size_t(currentSteps[path]), std::size_t(myRank)}, {1, 1}).write(value);
        }
        currentSteps[path] += 1;
    }

    // track vector values
    /**
     * @brief Track vector values (per MPI rank).
     *
     * @tparam T data set data type
     * @param path HDF5 *path* to data/key
     * @param steps
     * @param size vector size
     * @param maxSteps
     */
    template<typename T>
    void createVectorDataSet(const std::string& path, int steps, std::size_t size,
                             std::size_t maxSteps=HighFive::DataSpace::UNLIMITED){
        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking(std::vector<hsize_t>{1, 1, size}));
        dataSets[path] = h5file.createDataSet<T>(path, HighFive::DataSpace({std::size_t(steps),
                                                                            std::size_t(numProcs), size},
                                                                           {maxSteps, std::size_t(numProcs), size}),
                                                 props);
        vectorSizes[path] = size;
        currentSteps[path] = 0;
    }

    /**
     * Write value to vector value data set.
     *
     * @tparam T data set data type
     * @param path HDF5 *path* to data/key
     * @param data value(s)/data to be written as std::vector<T>
     */
    template<typename T>
    void vector2file(const std::string& path, std::vector<T> data){
        if (!disabled) {
            if (currentSteps[path] >= totalSteps[path]) {
                totalSteps[path] += 1;
                //Logger(INFO) << "resizing: " << totalSteps[path];
                dataSets[path].resize({std::size_t(totalSteps[path]), std::size_t(numProcs), vectorSizes[path]});
            }
            dataSets[path].select({std::size_t(currentSteps[path]), std::size_t(myRank), 0},
                                             {1, 1, vectorSizes[path]}).write(data);
        }
        currentSteps[path] += 1;
    }

    /**
     * Write value to vector value data set.
     *
     * @tparam T data set data type
     * @param path HDF5 *path* to data/key
     * @param data value(s)/data to be written as T*
     */
    template<typename T>
    void vector2file(const std::string& path, T *data) {
        if (!disabled) {
            std::vector<T> dataVector;
            for (int i=0; i<vectorSizes[path]; i++) {
                dataVector.push_back(data[i]);
            }
            vector2file(path, dataVector);
        }
    }

private:
    /**
     * Constructor.
     *
     * @param outfile output file
     */
    H5Profiler(const std::string& outfile);

    // basic containers for meta information
    HighFive::File h5file;
    std::unordered_map<std::string, HighFive::DataSet> dataSets;
    int numProcs;
    int step;
    int myRank;
    bool disabled;

    // timing variables
    std::unordered_map<std::string, double> timeStart;
    std::unordered_map<std::string, double> timeElapsed;

    std::unordered_map<std::string, int> currentSteps;
    std::unordered_map<std::string, int> totalSteps;
    // vector sizes
    std::unordered_map<std::string, std::size_t> vectorSizes;

};


#endif //MILUPHPC_H5PROFILER_H
