// Acknowledgment: Johannes Martin (GitHub: jammartin)
// changed to be resizable via HighFive::Chunking
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

namespace ProfilerIds {

    const char* const numParticles      { "/general/numParticles" };
    const char* const numParticlesLocal { "/general/numParticlesLocal" };
    const char* const ranges            { "/general/ranges" };

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

    }

}

// Singleton
class H5Profiler {
public:
    // constructor is only called ONCE due to the singleton pattern
    // arguments have to be passed when first calling getInstance()
    static H5Profiler& getInstance(const std::string& outfile = ""){
        static H5Profiler instance_(outfile);
        return instance_;
    }

    // deleting methods we don't want for the singleton design pattern (c++11)
    H5Profiler(H5Profiler const&) = delete;
    void operator=(H5Profiler const&) = delete;

    void const setStep(const int& _step) { step = _step; }
    void const setRank(const int& _myRank) { myRank = _myRank; }
    void const setNumProcs(const int& _numProcs) { numProcs = _numProcs; }
    void const disableWrite() { disabled = true; }
    void const enableWrite() { disabled = false; }

    /*void createTimeDataSet(const std::string& path, int steps);
    void time(const std::string &path);
    void timePause(const std::string &path);
    void time2file(const std::string &path, int myRank, bool onlyWrite=false);*/

    // TEMPLATE FUNCTIONS

    // track single values
    template<typename T>
    void createValueDataSet(const std::string& path, int steps, std::size_t maxSteps=HighFive::DataSpace::UNLIMITED){
        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking(std::vector<hsize_t>{1, 1}));
        dataSets[path] = h5file.createDataSet<T>(path, HighFive::DataSpace({std::size_t(steps), std::size_t(numProcs)},
                                                                        {maxSteps, std::size_t(numProcs)}), props);
        totalSteps[path] = steps;
    }

    template<typename T>
    void value2file(const std::string& path, T value) {

        if (!disabled) {
            if (step >= totalSteps[path]) {
                totalSteps[path] += 1;
                //Logger(INFO) << "resizing: " << totalSteps[path];
                dataSets[path].resize({std::size_t(totalSteps[path]), std::size_t(numProcs)});
            }
            dataSets[path].select({std::size_t(step), std::size_t(myRank)}, {1, 1}).write(value);
        }

    }

    // track vector values
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
    }

    template<typename T>
    void vector2file(const std::string& path, std::vector<T> data){
        if (!disabled) {
            if (step >= totalSteps[path]) {
                totalSteps[path] += 1;
                //Logger(INFO) << "resizing: " << totalSteps[path];
                dataSets[path].resize({std::size_t(totalSteps[path]), std::size_t(numProcs), vectorSizes[path]});
            }
            dataSets[path].select({std::size_t(step), std::size_t(myRank), 0},
                                             {1, 1, vectorSizes[path]}).write(data);
        }
    }

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

    std::unordered_map<std::string, int> totalSteps;
    // vector sizes
    std::unordered_map<std::string, std::size_t> vectorSizes;

};


#endif //MILUPHPC_H5PROFILER_H
