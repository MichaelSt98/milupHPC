#include "../include/miluphpc.h"
#include "../include/integrator/explicit_euler.h"
#include "../include/integrator/predictor_corrector_euler.h"
#include "../include/integrator/leapfrog.h"
#include "../include/integrator/godunov.h"
#include "../include/utils/config_parser.h"
#include <boost/filesystem.hpp>

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>

#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"

bool checkFile(const std::string file, bool terminate=false, const std::string message="");
bool checkFile(const std::string file, bool terminate, const std::string message) {
    std::ifstream fileStream(file.c_str());
    if (fileStream.good()) {
        return true;
    }
    else {
        if (terminate) {
            Logger(WARN) << message;
            Logger(ERROR) << "Provided file: " << file << " not available!";
            MPI_Finalize();
            exit(0);
        }
        return false;
    }
}

void SetDeviceBeforeInit()
{
    char * localRankStr = NULL;
    int rank = 0;
    int devCount = 2;

    if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL)
    {
        rank = atoi(localRankStr);
    }

    gpuErrorcheck(cudaGetDeviceCount(&devCount));
    //gpuErrorcheck(cudaSetDevice(rank % devCount));
    gpuErrorcheck(cudaSetDevice(rank % devCount));
}

structLog LOGCFG = {};

int main(int argc, char** argv)
{

    /// SETTINGS/INITIALIZATIONS
    // -----------------------------------------------------------------------------------------------------------------

    /// MPI rank setting
    SetDeviceBeforeInit();

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator comm;

    int rank = comm.rank();
    int numProcesses = comm.size();

    /// Setting CUDA device
    //cudaSetDevice(rank);
    int device;
    cudaGetDevice(&device);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    cudaDeviceSynchronize();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int numberOfMultiprocessors = deviceProp.multiProcessorCount;
    printf("numberOfMultiprocessors: %i\n", numberOfMultiprocessors);

    // testing whether MPI works ...
    //int mpi_test = rank;
    //all_reduce(comm, boost::mpi::inplace_t<int*>(&mpi_test), 1, std::plus<int>());
    //Logger(INFO) << "mpi_test = " << mpi_test;

    /// Command line argument parsing
    cxxopts::Options options("HPC NBody", "Multi-GPU CUDA Barnes-Hut NBody/SPH code");

    bool loadBalancing = false;

    options.add_options()
            ("n,number-output-files", "number of output files", cxxopts::value<int>()->default_value("100"))
            ("t,max-time-step", "time step", cxxopts::value<real>()->default_value("-1."))
            ("l,load-balancing", "load balancing", cxxopts::value<bool>(loadBalancing))
            ("L,load-balancing-interval", "load balancing interval", cxxopts::value<int>()->default_value("-1"))
            ("C,config", "config file", cxxopts::value<std::string>()->default_value("config/config.info"))
            ("m,material-config", "material config file", cxxopts::value<std::string>()->default_value("config/material.cfg"))
            ("c,curve-type", "curve type (Lebesgue: 0/Hilbert: 1)", cxxopts::value<int>()->default_value("-1"))
            ("f,input-file", "File name", cxxopts::value<std::string>()->default_value("-"))
            ("v,verbosity", "Verbosity level", cxxopts::value<int>()->default_value("0"))
            ("h,help", "Show this help");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (cxxopts::OptionException &exception) {
        std::cerr << exception.what();
        MPI_Finalize();
        exit(0);
    }
    catch (...) {
        std::cerr << "Error parsing ...";
        MPI_Finalize();
        exit(0);
    }

    if (result.count("help")) {
        if (rank == 0) {
            std::cout << options.help() << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }

    /// Config file parsing
    std::string configFile = result["config"].as<std::string>();
    checkFile(configFile, true, std::string{"Provided config file not available!"});

    /// Collect settings/information in struct
    SimulationParameters parameters;

    parameters.verbosity = result["verbosity"].as<int>();
    /// Logger settings
    LOGCFG.headers = true;
    if (parameters.verbosity == 0) {
        LOGCFG.level = TRACE;
    }
    else if (parameters.verbosity == 1) {
        LOGCFG.level = INFO;
    }
    else if (parameters.verbosity >= 2) {
        LOGCFG.level = DEBUG;
    }
    else {
        LOGCFG.level = TRACE;
    }
    //LOGCFG.level = static_cast<typeLog>(1); //TRACE; //DEBUG;
    LOGCFG.rank = rank;
    //LOGCFG.outputRank = 0;
    //Logger(DEBUG) << "DEBUG output";
    //Logger(WARN) << "WARN output";
    //Logger(ERROR) << "ERROR output";
    //Logger(INFO) << "INFO output";
    //Logger(TRACE) << "TRACE output";
    //Logger(TIME) << "TIME output";

    Logger(DEBUG) << "rank: " << rank << " | number of processes: " << numProcesses;
    Logger(DEBUG) << "device: " << device << " | num devices: " << numDevices;

    Logger(INFO) << "sizeof(keyType) = " << (int)sizeof(keyType); // intmax_t, uintmax_t
    Logger(INFO) << "sizeof(intmax_t) = " << (int)sizeof(intmax_t);
    Logger(INFO) << "sizeof(uintmax_t) = " << (int)sizeof(uintmax_t); // __int128
    Logger(INFO) << "sizeof(__int128) = " << (int)sizeof(__int128);

    ConfigParser confP{ConfigParser(configFile.c_str())}; //"config/config.info"
    LOGCFG.write2LogFile = confP.getVal<bool>("log");
    LOGCFG.omitTime = confP.getVal<bool>("omitTime");

    parameters.directory = confP.getVal<std::string>("directory");
    if (boost::filesystem::create_directories(parameters.directory)) {
        Logger(DEBUG) << "Created directory: " << parameters.directory;
    }
    parameters.logDirectory = parameters.directory + std::string{"log/"};
    if (boost::filesystem::create_directories(parameters.logDirectory)) {
        Logger(DEBUG) << "Created directory: " << parameters.logDirectory;
    }

    std::stringstream logFileName;
    logFileName << parameters.logDirectory << "miluphpc.log";
    LOGCFG.logFileName = logFileName.str();
    Logger(TRACE) << "log file to: " << logFileName.str();
    //MPI_Finalize();
    //exit(0);

    parameters.timeStep = confP.getVal<real>("timeStep");
    parameters.maxTimeStep = result["max-time-step"].as<real>();
    if (parameters.maxTimeStep < 0.) {
        parameters.maxTimeStep = confP.getVal<real>("maxTimeStep");
    }
    parameters.timeEnd = confP.getVal<real>("timeEnd");
    parameters.outputRank = confP.getVal<int>("outputRank");
    if (parameters.outputRank < 0 || parameters.outputRank >= numProcesses) {
        parameters.outputRank = -1; // if selected output rank is not valid, log all processes
    }
    LOGCFG.outputRank = parameters.outputRank;
    parameters.performanceLog = confP.getVal<bool>("performanceLog");
    parameters.particlesSent2H5 = confP.getVal<bool>("particlesSent2H5");
    parameters.sfcSelection = confP.getVal<int>("sfc");
    if (result["curve-type"].as<int>() != -1) {
        parameters.sfcSelection = result["curve-type"].as<int>();
    }
    parameters.integratorSelection = confP.getVal<int>("integrator");
//#if GRAVITY_SIM
    parameters.theta = confP.getVal<real>("theta");
    parameters.smoothing = confP.getVal<real>("smoothing");
    parameters.gravityForceVersion = confP.getVal<int>("gravityForceVersion");
//#endif
//#if SPH_SIM
    parameters.smoothingKernelSelection = confP.getVal<int>("smoothingKernel");
    parameters.sphFixedRadiusNNVersion = confP.getVal<int>("sphFixedRadiusNNVersion");
//#endif
#if MESHLESS_FINITE_METHOD
    parameters.defaultRiemannSolver = confP.getVal<int>("defaultRiemannSolver");
#else
    parameters.defaultRiemannSolver = 0; // dummy value as unused anyway
#endif
    parameters.removeParticles = confP.getVal<bool>("removeParticles");
    parameters.removeParticlesCriterion = confP.getVal<int>("removeParticlesCriterion");
    parameters.removeParticlesDimension = confP.getVal<real>("removeParticlesDimension");
    parameters.numOutputFiles = result["number-output-files"].as<int>();
    parameters.timeKernels = true;
    parameters.loadBalancing = confP.getVal<bool>("loadBalancing");
    if (parameters.loadBalancing || loadBalancing) {
        parameters.loadBalancing = true;
    }
    parameters.loadBalancingInterval = confP.getVal<int>("loadBalancingInterval");
    if (result["load-balancing-interval"].as<int>() > 0) {
        parameters.loadBalancingInterval = result["load-balancing-interval"].as<int>();
    }
    parameters.loadBalancingBins = confP.getVal<int>("loadBalancingBins");
    parameters.verbosity = result["verbosity"].as<int>();
    parameters.materialConfigFile = result["material-config"].as<std::string>();
    parameters.inputFile = result["input-file"].as<std::string>();
    parameters.particleMemoryContingent = confP.getVal<real>("particleMemoryContingent");
    if (parameters.particleMemoryContingent > 1.0 || parameters.particleMemoryContingent < 0.0) {
        parameters.particleMemoryContingent = 1.0;
        Logger(WARN) << "Setting particle memory contingent to: " << parameters.particleMemoryContingent;
    }
    parameters.calculateCenterOfMass = confP.getVal<bool>("calculateCenterOfMass");
    //TODO: apply those
    parameters.calculateAngularMomentum = confP.getVal<bool>("calculateAngularMomentum");
    parameters.calculateEnergy = confP.getVal<bool>("calculateEnergy");

    // + 1 should not be necessary, but need to investigate whether this is a problem for 1 GPU sims
    parameters.domainListSize = POW_DIM * MAX_LEVEL * (numProcesses - 1) + 1;
    Logger(DEBUG) << "domainListSize: " << parameters.domainListSize;

#if MESHLESS_FINITE_METHOD
    parameters.critCondNum = confP.getVal<real>("critCondNum");
    parameters.betaMin = confP.getVal<real>("betaMin");
    parameters.betaMax = confP.getVal<real>("betaMax");
#if PAIRWISE_LIMITER
    parameters.psi1 = confP.getVal<real>("psi1");
    parameters.psi2 = confP.getVal<real>("psi2");
#endif // PAIRWISE_LIMITER
#endif // MESHLESS_FINITE_METHOD

    if (!checkFile(parameters.materialConfigFile, false)) {
        parameters.materialConfigFile = std::string{"config/material.cfg"};
        checkFile(parameters.materialConfigFile, true,
                  std::string{"Provided material config file and default (config/material.cfg) not available!"});
    }
    checkFile(parameters.inputFile, true, std::string{"Provided input file not available!"});


    /// H5 profiling/profiler
    // profiling
    std::stringstream profilerFile;
    profilerFile << parameters.logDirectory << "performance.h5";
    H5Profiler &profiler = H5Profiler::getInstance(profilerFile.str());
    profiler.setRank(comm.rank());
    profiler.setNumProcs(comm.size());
    if (!parameters.performanceLog) {
        profiler.disableWrite();
    }
    // General
    profiler.createValueDataSet<int>(ProfilerIds::numParticles, 1);
    profiler.createValueDataSet<int>(ProfilerIds::numParticlesLocal, 1);
    profiler.createVectorDataSet<keyType>(ProfilerIds::ranges, 1, comm.size() + 1);
    // Timing
    profiler.createValueDataSet<real>(ProfilerIds::Time::rhs, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::rhsElapsed, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::loadBalancing, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::reset, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::removeParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::boundingBox, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::assignParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::tree, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::pseudoParticle, 1);
#if GRAVITY_SIM
    profiler.createValueDataSet<real>(ProfilerIds::Time::gravity, 1);
    profiler.createVectorDataSet<int>(ProfilerIds::SendLengths::gravityParticles, 1, numProcesses);
    profiler.createVectorDataSet<int>(ProfilerIds::SendLengths::gravityPseudoParticles, 1, numProcesses);
    profiler.createVectorDataSet<int>(ProfilerIds::ReceiveLengths::gravityParticles, 1, numProcesses);
    profiler.createVectorDataSet<int>(ProfilerIds::ReceiveLengths::gravityPseudoParticles, 1, numProcesses);
#endif
#if SPH_SIM
    profiler.createValueDataSet<real>(ProfilerIds::Time::sph, 1);
    profiler.createVectorDataSet<int>(ProfilerIds::SendLengths::sph, 1, numProcesses);
    profiler.createVectorDataSet<int>(ProfilerIds::ReceiveLengths::sph, 1, numProcesses);
#endif
    // Detailed timing
    profiler.createValueDataSet<real>(ProfilerIds::Time::Tree::createDomain, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Tree::tree, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Tree::buildDomain, 1);
#if GRAVITY_SIM
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::compTheta, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::symbolicForce, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::sending, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::insertReceivedPseudoParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::insertReceivedParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::force, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::repairTree, 1);
#endif
#if SPH_SIM
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::compTheta, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::determineSearchRadii, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::symbolicForce, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::sending, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::insertReceivedParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::fixedRadiusNN, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::soundSpeed, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::pressure, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::resend, 1);
#if MESHLESS_FINITE_METHOD
    profiler.createValueDataSet<real>(ProfilerIds::Time::MFV::density, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::MFV::vectorWeights, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::MFV::gradients, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::MFV::riemannFluxes, 1);
#else
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::density, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::internalForces, 1);
#endif
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::repairTree, 1);
#endif
    profiler.createValueDataSet<real>(ProfilerIds::Time::integrate, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::IO, 1);


    /// INTEGRATOR SELECTION
    // -----------------------------------------------------------------------------------------------------------------
    Miluphpc *miluphpc;
    // miluphpc = new Miluphpc(parameters, numParticles, numNodes); // not possible since abstract class
    switch (parameters.integratorSelection) {
        case IntegratorSelection::explicit_euler: {
            miluphpc = new ExplicitEuler(parameters);
        } break;
        case IntegratorSelection::predictor_corrector_euler: {
            miluphpc = new PredictorCorrectorEuler(parameters);
        } break;
        case IntegratorSelection::leapfrog: {
            miluphpc = new Leapfrog(parameters);
        } break;
        case IntegratorSelection::godunov: {
#if !MESHLESS_FINITE_METHOD
            Logger(ERROR) << "Godunov type integrator must not be used with classic SPH. Use MFV/MFM instead. - Aborting.";
            MPI_Finalize();
            exit(2);
#endif
            miluphpc = new Godunov(parameters);
        } break;
        default: {
            Logger(ERROR) << "Integrator not available!";
            MPI_Finalize();
            exit(1);
        }
    }

    if (rank == 0) {
        Logger(TRACE) << "---------------STARTING---------------";
    }

    Timer timer;
    real timeElapsed;
    /// MAIN LOOP
    // -----------------------------------------------------------------------------------------------------------------
    real t = 0;
    for (int i_step=0; i_step<parameters.numOutputFiles; i_step++) {

        //profiler.setStep(i_step);

        Logger(TRACE) << "-----------------------------------------------------------------";
        Logger(TRACE, true) << "STEP: " << i_step;
        Logger(TRACE) << "-----------------------------------------------------------------";

        *miluphpc->simulationTimeHandler->h_subEndTime += (parameters.timeEnd/(real)parameters.numOutputFiles);
        Logger(DEBUG) << "subEndTime += " << (parameters.timeEnd/(real)parameters.numOutputFiles);

        miluphpc->simulationTimeHandler->copy(To::device);

        miluphpc->integrate(i_step);
        miluphpc->afterIntegrationStep();

        timer.reset();
        auto time = miluphpc->particles2file(i_step);
        timeElapsed = timer.elapsed();
        Logger(TIME) << "particles2file: " << timeElapsed << " ms";
        profiler.value2file(ProfilerIds::Time::IO, timeElapsed);


        t += parameters.timeStep;

    }

    /// END OF SIMULATION
    // -----------------------------------------------------------------------------------------------------------------
    comm.barrier();
    LOGCFG.outputRank = -1;
    if (rank == 0) {
        Logger(TRACE) << "\n\n";
        Logger(TRACE) << "---------------FINISHED---------------";
        Logger(TRACE) << "Input file: " << parameters.inputFile;
        Logger(TRACE) << "Config file: " << parameters.materialConfigFile;
        Logger(TRACE) << "Material config: " << parameters.materialConfigFile;
        if (LOGCFG.write2LogFile) {
            Logger(TRACE) << "Log file saved to: " << LOGCFG.logFileName;
        }
        if (parameters.performanceLog) {
            Logger(TRACE) << "Performance log saved to " << profilerFile.str();
        }
        if (parameters.particlesSent2H5) {
            Logger(TRACE) << "(Most recent) particles sent saved to: " << parameters.logDirectory;
        }
        Logger(TRACE) << "Generated " << parameters.numOutputFiles << " files!";
        Logger(TRACE) << "Data saved to " << parameters.directory;
        Logger(TRACE) << "---------------FINISHED---------------";
    }

    return 0;
}