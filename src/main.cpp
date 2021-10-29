#include "../include/miluphpc.h"
#include "../include/integrator/euler.h"
#include "../include/integrator/explicit_euler.h"
#include "../include/integrator/predictor_corrector.h"
#include "../include/integrator/predictor_corrector_euler.h"
#include "../include/utils/config_parser.h"

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>

#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"

void SetDeviceBeforeInit()
{
    char * localRankStr = NULL;
    int rank = 0, devCount = 2;

    // We extract the local rank initialization using an environment variable
    if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL)
    {
        rank = atoi(localRankStr);
    }
    Logger(INFO) << "devCount: " << devCount << " | rank: " << rank
    << " | setting device: " << rank % devCount;
    safeCudaCall(cudaGetDeviceCount(&devCount));
    safeCudaCall(cudaSetDevice(rank % devCount));
}

structlog LOGCFG = {};

int main(int argc, char** argv)
{

    //SetDeviceBeforeInit();

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator comm;

    int rank = comm.rank();
    int numProcesses = comm.size();

    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;
    LOGCFG.myrank = rank;

    printf("Hello World from proc %i out of %i\n", rank, numProcesses);

    cudaSetDevice(rank);
    int device;
    cudaGetDevice(&device);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    Logger(INFO) << "device: " << device << " | num devices: " << numDevices;

    cudaDeviceSynchronize();

    int mpi_test = rank;
    all_reduce(comm, boost::mpi::inplace_t<int*>(&mpi_test), 1, std::plus<int>());
    Logger(INFO) << "mpi_test = " << mpi_test;

    ConfigParser confP{ConfigParser("config/config.info")};
    real timeStep = confP.getVal<real>("timeStep");
    Logger(INFO) << "timeStep from config file: " << timeStep;

    cxxopts::Options options("HPC NBody", "Multi-GPU CUDA Barnes-Hut NBody/SPH code");

    bool render = false;
    bool loadBalancing = false;

    options.add_options()
            ("i,iterations", "number of iterations", cxxopts::value<int>()->default_value("100"))
            //("n,particles", "number of particles", cxxopts::value<int>()->default_value("524288")) //"524288"
            ("t,timestep", "time step", cxxopts::value<float>()->default_value("0.001"))
            ("l,loadbalancing", "load balancing", cxxopts::value<bool>(loadBalancing))
            ("L,loadbalancinginterval", "load balancing interval", cxxopts::value<int>()->default_value("10"))
            //("m,material", "material config file", cxxopts::value<std::string>()->default_value("config/material.cfg"))
            ("c,curvetype", "curve type (Lebesgue: 0/Hilbert: 1)", cxxopts::value<int>()->default_value("0"))
            ("f,filename", "File name", cxxopts::value<std::string>()->default_value("-"))
            //("v,verbosity", "Verbosity level")
            ("h,help", "Show this help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    /** Initialization */
    SimulationParameters parameters;

    parameters.iterations = result["iterations"].as<int>(); //500;
    //parameters.numberOfParticles = result["particles"].as<int>(); //512*256*4;
    parameters.timestep = result["timestep"].as<float>();
    parameters.dampening = 1.0;
    parameters.timeKernels = true;
    parameters.loadBalancing = loadBalancing;
    parameters.loadBalancingInterval = result["loadbalancinginterval"].as<int>();
    //parameters.loadBalancing = loadBalancing;
    //parameters.loadBalancingInterval = result["loadbalancinginterval"].as<int>();
    parameters.curveType = result["curvetype"].as<int>();
    std::string filename = result["filename"].as<std::string>();
    parameters.filename = filename;

    parameters.sml = confP.getVal<real>("sml");

    //LOGCFG.outputRank = 0;
    //Logger(DEBUG) << "DEBUG output";
    //Logger(WARN) << "WARN output";
    //Logger(ERROR) << "ERROR output";
    //Logger(INFO) << "INFO output";
    //Logger(TIME) << "TIME output";

#if DEBUGGING
#ifdef SINGLE_PRECISION
    Logger(INFO) << "typedef float real";
#else
    Logger(INFO) << "typedef double real";
#endif
#endif

    IntegratorSelection::Type integratorSelection = IntegratorSelection::explicit_euler;
    //IntegratorSelection::Type integratorSelection = IntegratorSelection::predictor_corrector_euler;

    Miluphpc *miluphpc;
    // miluphpc = new Miluphpc(parameters, numParticles, numNodes); // not possible since abstract class

    switch (integratorSelection) {
        case IntegratorSelection::explicit_euler: {
            miluphpc = new ExplicitEuler(parameters);
        } break;
        case IntegratorSelection::euler: {
            miluphpc = new Euler(parameters);
        } break;
        case IntegratorSelection::predictor_corrector: {
            miluphpc = new PredictorCorrector(parameters);
        } break;
        case IntegratorSelection::predictor_corrector_euler: {
            miluphpc = new PredictorCorrectorEuler(parameters);
        } break;
        default: {
            printf("Integrator not available!");
        }
    }

    // profiling
    H5Profiler &profiler = H5Profiler::getInstance("log/performance.h5");
    profiler.setRank(comm.rank());
    profiler.setNumProcs(comm.size());

    profiler.createValueDataSet<int>(ProfilerIds::numParticles, 1);
    profiler.createValueDataSet<int>(ProfilerIds::numParticlesLocal, 1);
    profiler.createVectorDataSet<keyType>(ProfilerIds::ranges, 1, comm.size() + 1);

    profiler.createValueDataSet<real>(ProfilerIds::Time::rhs, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::rhsElapsed, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::loadBalancing, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::reset, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::boundingBox, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::assignParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::tree, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::pseudoParticle, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::gravity, 1);
#if SPH_SIM
    profiler.createValueDataSet<real>(ProfilerIds::Time::sph, 1);
#endif
    profiler.createValueDataSet<real>(ProfilerIds::Time::Tree::createDomain, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Tree::tree, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Tree::buildDomain, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::compTheta, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::symbolicForce, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::sending, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::insertReceivedPseudoParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::insertReceivedParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::force, 1);

    int fileCounter = 0;
    for (int i_step=0; i_step<parameters.iterations; i_step++) {

        profiler.setStep(i_step);

        Logger(INFO) << "-----------------------------------------------------------------";
        Logger(INFO) << "STEP: " << i_step;
        Logger(INFO) << "-----------------------------------------------------------------";

        miluphpc->integrate(i_step);

        if (i_step == 0 || i_step % 10 == 0) {
            auto time = miluphpc->particles2file(fileCounter);
            fileCounter++;
            Logger(TIME) << "particles2file: " << time << " ms";
        }

    }

    Logger(INFO) << "---------------FINISHED---------------";

    return 0;
}