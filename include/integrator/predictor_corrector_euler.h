/**
 * @file predictor_corrector_euler.h
 * @brief Predictor Corrector Euler integrator.
 *
 * Predictor Corrector Euler integrator inheriting from the Miluphpc class.
 *
 * Predictor corrector integrators generally consist out of one or multiple predictor steps and a corresponding
 * correction. Since the accelerations needed for correction are determined using the predicted positions of the
 * particles it may happen that particles are moving out of their current domain and do not belong anymore to the
 * same process they belonged to before the prediction step. By the construction of the presented parallelization
 * approach, this would require a subsequent reassigning of those particles. However, this includes the particle
 * exchange between the MPI processes and consequently an inevitable resorting of all particles. Since the information
 * need to be recombined for each particle in order to complete the integration step for the predictor-corrector scheme,
 * the particles would have to be sent back and correctly ordered to align the original particle sorting. This is
 * surely doable but most likely really inefficient in terms of performance.
 *
 * The reason for the necessity of having the particles on the process they belong to is the correctness of the
 * parallel tree. More precisely the correctness of the pseudo-particles and their COMs and masses, which is a key
 * ingredient of the Barnes-Hut method for calculating the gravitational forces.
 *
 * However, the SPH part is not relying on the correctness of the pseudo-particles and the neighbor search
 * using the tree can be performed correctly as long as the pseudo-particle center-of-masses are within the correct
 * cell regarding their location in the tree. Hence, by decoupling gravity it is possible to use the
 * predictor-corrector scheme for SPH only and allow a temporarily corrupt tree in means of false pseudo-particles
 * avoiding the necessity of assigning the particles to the correct process for the corrector step.
 *
 * Though, for finding the correct interaction partners for each particle it is necessary to have an
 * additional constraint on the chosen time step. Particles should not move farther than the half of the
 * search radius used to find the particles to be exchanged.
 *
 * @author Michael Staneker
 * @bug no known bugs
 */
#ifndef MILUPHPC_PREDICTOR_CORRECTOR_EULER_H
#define MILUPHPC_PREDICTOR_CORRECTOR_EULER_H

#include "../miluphpc.h"
#include "device_predictor_corrector_euler.cuh"
#include "device_explicit_euler.cuh"


class PredictorCorrectorEuler : public Miluphpc {

private:

public:

    int device;
#if TARGET_GPU
    struct cudaDeviceProp prop;
    int *d_blockCount;

    real *d_block_forces;
    real *d_block_courant;
    real *d_block_artVisc;
    real *d_block_e;
    real *d_block_rho;
    real *d_block_vmax;

    PredictorCorrectorEulerNS::BlockShared *d_blockShared;
#endif // TARGET_GPU
    /**
     * @brief Constructor.
     *
     * @param simulationParameters Simulation parameters/settings.
     */
    PredictorCorrectorEuler(SimulationParameters simulationParameters);

    /**
     * @brief Destructor.
     */
    ~PredictorCorrectorEuler();

    /**
     * @brief Implementation of the abstract integration method.
     *
     * @param step Integration step (number)
     */
    void integrate(int step);
};

#endif //MILUPHPC_PREDICTOR_CORRECTOR_EULER_H
