/*#include "../include/constants.h"
#include "../include/utils/logger.h"
#include "../include/utils/timer.h"
#include "../include/utils/cxxopts.h"
#include "../include/utils/config_parser.h"
#include "../include/cuda_utils/cuda_utilities.cuh"
//#include "../include/cuda_utils/cuda_launcher.cuh"

#include "../include/subdomain_key_tree/tree.cuh"
#include "../include/particles.cuh"
#include "../include/particle_handler.h"
#include "../include/device_rhs.cuh"
#include "../include/subdomain_key_tree/subdomain_handler.h"*/

#include "../include/miluphpc.h"
#include "../include/integrator/euler.h"
#include "../include/integrator/explicit_euler.h"
#include "../include/integrator/predictor_corrector.h"

//#include <mpi.h>
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

    printf("Hello World from proc %i out of %i\n", rank, numProcesses);

    //cudaSetDevice(rank);
    int device;
    cudaGetDevice(&device);
    Logger(INFO) << "Set device to " << device;

    cxxopts::Options options("HPC NBody", "Multi-GPU CUDA Barnes-Hut NBody code");

    bool render = false;
    bool loadBalancing = false;

    options.add_options()
            //("r,render", "render simulation", cxxopts::value<bool>(render))
            ("i,iterations", "number of iterations", cxxopts::value<int>()->default_value("100"))
            ("n,particles", "number of particles", cxxopts::value<int>()->default_value("524288")) //"524288"
            //("b,blocksize", "block size", cxxopts::value<int>()->default_value("256"))
            //("g,gridsize", "grid size", cxxopts::value<int>()->default_value("1024"))
            //("R,renderinterval", "render interval", cxxopts::value<int>()->default_value("10"))
            //("l,loadbalancing", "load balancing", cxxopts::value<bool>(loadBalancing))
            //("L,loadbalancinginterval", "load balancing interval", cxxopts::value<int>()->default_value("10"))
            ("m,material", "material config file", cxxopts::value<std::string>()->default_value("config/material.cfg"))
            ("c,curvetype", "curve type (Lebesgue/Hilbert)", cxxopts::value<int>()->default_value("0"))
            ("v,verbosity", "Verbosity level")
            ("h,help", "Show this help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    /** Initialization */
    SimulationParameters parameters;

    parameters.iterations = result["iterations"].as<int>(); //500;
    parameters.numberOfParticles = result["particles"].as<int>(); //512*256*4;
    parameters.timestep = 0.001;
    parameters.gravity = 1.0;
    parameters.dampening = 1.0;
    //parameters.gridSize = result["gridsize"].as<int>(); //1024;
    //parameters.blockSize = result["blocksize"].as<int>(); //256;
    //parameters.warp = 32;
    //parameters.stackSize = 64;
    //parameters.renderInterval = result["renderinterval"].as<int>(); //10;
    parameters.timeKernels = true;
    //parameters.loadBalancing = loadBalancing;
    //parameters.loadBalancingInterval = result["loadbalancinginterval"].as<int>();
    parameters.curveType = result["curvetype"].as<int>();

    LOGCFG.headers = true;
    LOGCFG.level = DEBUG;
    LOGCFG.myrank = rank;
    //LOGCFG.outputRank = 0;

    Logger(DEBUG) << "DEBUG output";
    Logger(WARN) << "WARN output";
    Logger(ERROR) << "ERROR output";
    Logger(INFO) << "INFO output";
    Logger(TIME) << "TIME output";

    integer numParticles = 100000;
    integer numNodes = 2 * numParticles + 50000; //12000;

    //integer numParticles = 7500;
    //integer numNodes = 3 * numParticles + 12000;

    //IntegratorSelection::Type integratorSelection = IntegratorSelection::euler;
    IntegratorSelection::Type integratorSelection = IntegratorSelection::explicit_euler;

    Miluphpc *miluphpc;
    switch (integratorSelection) {
        case IntegratorSelection::explicit_euler: {
            miluphpc = new ExplicitEuler(numParticles, numNodes);
        } break;
        case IntegratorSelection::euler: {
            miluphpc = new Euler(numParticles, numNodes);
        } break;
        case IntegratorSelection::predictor_corrector: {
            miluphpc = new PredictorCorrector(numParticles, numNodes);
        } break;
        default: {
            printf("Integrator not available!");
        }
    }

    miluphpc->loadDistribution();

    for (int i=0; i<parameters.iterations; i++) {

        Logger(INFO) << "-----------------------------------------------------------------";
        Logger(INFO) << "STEP: " << i;
        Logger(INFO) << "-----------------------------------------------------------------";

        std::stringstream stepss;
        stepss << std::setw(6) << std::setfill('0') << i;

        HighFive::File h5file("output/ts" + stepss.str() + ".h5",
                              HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate,
                              HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

        std::vector <size_t> dataSpaceDims(2);
        dataSpaceDims[0] = std::size_t(numParticles);
        dataSpaceDims[1] = 3;

        HighFive::DataSet ranges = h5file.createDataSet<keyType>("/hilbertRanges",
                                                                 HighFive::DataSpace(numProcesses+1));

        keyType *rangeValues;
        rangeValues = new keyType[numProcesses+1];


        //miluphpc.subDomainKeyTreeHandler->toHost();
        miluphpc->subDomainKeyTreeHandler->copy(To::host, true, false);
        for (int i=0; i<numProcesses+1; i++) {
            rangeValues[i] = miluphpc->subDomainKeyTreeHandler->h_subDomainKeyTree->range[i];
            Logger(INFO) << "rangeValues[" << i << "] = " << rangeValues[i];
        }

        ranges.write(rangeValues);

        delete [] rangeValues;

        HighFive::DataSet pos = h5file.createDataSet<real>("/x", HighFive::DataSpace(dataSpaceDims));
        HighFive::DataSet vel = h5file.createDataSet<real>("/v", HighFive::DataSpace(dataSpaceDims));
        HighFive::DataSet key = h5file.createDataSet<keyType>("/hilbertKey",
                                                              HighFive::DataSpace(numParticles));
        //miluphpc.run();
        //miluphpc.barnesHut();
        //miluphpc.sph();
        miluphpc->integrate();

        auto time = miluphpc->particles2file(&pos, &vel, &key);
        Logger(TIME) << "particles2file: " << time << " ms";

    }

    Logger(INFO) << "Finished!";

    return 0;
}