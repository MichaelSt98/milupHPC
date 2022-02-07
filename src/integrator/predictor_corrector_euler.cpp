#include "../../include/integrator/predictor_corrector_euler.h"

PredictorCorrectorEuler::PredictorCorrectorEuler(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {
    printf("PredictorCorrectorEuler()\n");
    integratedParticles = new IntegratedParticleHandler(numParticles, numNodes);


    cudaGetDevice(&device);
    printf("Device: %i\n", device);
    cudaGetDeviceProperties(&prop, device);
    //printf("prop.multiProcessorCount: %i\n", prop.multiProcessorCount);
    cuda::malloc(d_blockCount, 1);
    cuda::set(d_blockCount, 0, 1);

    cuda::malloc(d_block_forces, prop.multiProcessorCount);
    cuda::malloc(d_block_courant, prop.multiProcessorCount);
    cuda::malloc(d_block_artVisc, prop.multiProcessorCount);
    cuda::malloc(d_block_e, prop.multiProcessorCount);
    cuda::malloc(d_block_rho, prop.multiProcessorCount);
    cuda::malloc(d_block_vmax, prop.multiProcessorCount);

    cuda::malloc(d_blockShared, 1);

    PredictorCorrectorEulerNS::BlockSharedNS::Launch::set(d_blockShared, d_block_forces, d_block_courant,
                                                  d_block_courant);
    PredictorCorrectorEulerNS::BlockSharedNS::Launch::setE(d_blockShared, d_block_e);
    PredictorCorrectorEulerNS::BlockSharedNS::Launch::setRho(d_blockShared, d_block_rho);
    PredictorCorrectorEulerNS::BlockSharedNS::Launch::setVmax(d_blockShared, d_block_vmax);

}

PredictorCorrectorEuler::~PredictorCorrectorEuler() {
    printf("~PredictorCorrectorEuler()\n");

    delete [] integratedParticles;

    cuda::free(d_block_forces);
    cuda::free(d_block_courant);
    cuda::free(d_block_artVisc);
    cuda::free(d_block_e);
    cuda::free(d_block_rho);
    cuda::free(d_block_vmax);

    cuda::free(d_blockShared);
}




