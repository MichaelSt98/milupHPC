/*#include "../include/constants.h"
#include "../include/utils/logger.h"
#include "../include/utils/timer.h"
#include "../include/utils/cxxopts.h"
#include "../include/utils/config_parser.h"
#include "../include/cuda_utils/cuda_utilities.cuh"
//#include "../include/cuda_utils/cuda_launcher.cuh"

#include "../include/subdomain_key_tree/tree.cuh"
#include "../include/subdomain_key_tree/sample.cuh"
#include "../include/particles.cuh"
#include "../include/particle_handler.h"
#include "../include/memory_handling.h"
#include "../include/device_rhs.cuh"
#include "../include/subdomain_key_tree/subdomain_handler.h"*/

#include "../include/rhs.h"
#include "../include/miluphpc.h"

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

    MPI_Init(&argc, &argv);

    int rank;
    int numProcesses;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

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

    // ...

    integer numParticles = 100000;
    integer numNodes = 3 * numParticles + 12000;

    Miluphpc miluphpc(numParticles, numNodes);
    miluphpc.run();


    //ParticleHandler particleHandler(1000, 2000);
    //particleHandler.h_particles->x[0] = 10.f;
    //Logger(INFO) << "particleHandler.h_particles->x[0] = " << particleHandler.h_particles->x[0];



    Logger(INFO) << "Finished!";

    MPI_Finalize();
    return 0;
}


/*
 // host allocation/declaration
    int *test = new int[5];
    for (int i=0; i<5; i++) {
        test[i] = 0;
    }
    Foo *foo = new Foo(); //new Foo(test);
    foo->aMethod(test);
    for (int i=0; i<5; i++) {
        Logger(INFO) << "foo->d_test[" << i << "] = " << foo->d_test[i];
    }

    // device allocation/declaration
    Foo *d_foo;
    const size_t sz = sizeof(Foo);
    int *d_test;
    cudaMalloc((void**)&d_test, 5 * sizeof(int));
    cudaMalloc((void**)&d_foo, sz);
    //set d_test as member for d_foo
    launchSetKernel(d_foo, d_test);
    gpuErrorcheck( cudaPeekAtLastError() );
    gpuErrorcheck( cudaDeviceSynchronize() );
    // launch a kernel
    launchTestKernel(d_foo);
    gpuErrorcheck( cudaPeekAtLastError() );
    gpuErrorcheck( cudaDeviceSynchronize() );

    //gpuErrorcheck(cudaMemcpy(foo, d_foo, sizeof(Foo) + 5 * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorcheck(cudaMemcpy(foo->d_test, d_test, 5 * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i=0; i<5; i++) {
    Logger(INFO) << "foo->d_test[" << i << "] = " << foo->d_test[i];
    }

    delete [] test;
    gpuErrorcheck( cudaFree(d_test) );
    gpuErrorcheck( cudaFree(d_foo) );
 */
