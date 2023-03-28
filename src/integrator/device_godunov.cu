#include "../../include/integrator/device_godunov.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

__global__ void GodunovNS::Kernel::selectTimestep(SimulationTime *simulationTime, Particles *particles, int numParticles,
                                                  real *dtBlockShared, int *blockCount){

//#define SAFETY_FIRST 0.1

    __shared__ real sharedTimestep[NUM_THREADS_LIMIT_TIME_STEP]; // timestep

    int i, j, ip, noi, d, k, m; // loop variables
    real dt, temp, dx[DIM], dv[DIM], dxAbs;
    real signalVel = DBL_MIN;

    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x) {
        noi = particles->noi[i];

        for (j = 0; j<noi; j++){

            ip = particles->nnl[i*MAX_NUM_INTERACTIONS+j];

            dx[0] = particles->x[i] - particles->x[ip];
            dv[0] = particles->vx[i] - particles->vx[ip];
            dxAbs = dx[0]*dx[0];
#if DIM > 1
            dx[1] = particles->y[i] - particles->y[ip];
            dv[1] = particles->vy[i] - particles->vy[ip];
            dxAbs += dx[1]*dx[1];
#if DIM == 3
            dx[2] = particles->z[i] - particles->z[ip];
            dv[2] = particles->vz[i] - particles->vz[ip];
            dxAbs += dx[2]*dx[2];
#endif
#endif
            dxAbs = sqrt(dxAbs);

            temp = 0.;
#pragma unroll
            for (d=0; d<DIM; d++){
                temp += dv[d]*dx[d];
            }

            temp = cuda::math::min(0., temp/dxAbs);
            signalVel = cuda::math::max(signalVel, particles->cs[i] + particles->cs[ip] - temp);
        }

        dt = 2.*COURANT_FACT*particles->sml[i]/abs(signalVel);
        //TODO: factor 2 allows for quite a large timestep
        //dt = COURANT_FACT*particles->sml[i]/signalVel;
        //printf("Selected timestep dt = %e\n", dt);
    }
    __threadfence();

    i = threadIdx.x;
    sharedTimestep[i] = dt;

    for (j = NUM_THREADS_LIMIT_TIME_STEP / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            sharedTimestep[i] = dt = cuda::math::min(dt, sharedTimestep[k]);
        }
    }
    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        dtBlockShared[k] = dt;

        m = gridDim.x - 1;
        if (m == atomicInc((unsigned int *) blockCount, m)) {
            // last block, so combine all block results
            for (j = 0; j <= m; j++) {
                dt = cuda::math::min(dt, dtBlockShared[j]);
            }

            // select timestep
            *simulationTime->dt = dt;

            *simulationTime->dt = cuda::math::min(*simulationTime->dt,
                                                  *simulationTime->subEndTime - *simulationTime->currentTime);
            if (*simulationTime->dt > *simulationTime->dt_max) {
                *simulationTime->dt = *simulationTime->dt_max;
            }

            // reset block count
            *blockCount = 0;
        }
    }
}

real GodunovNS::Kernel::Launch::selectTimestep(int multiProcessorCount, SimulationTime *simulationTime, Particles *particles,
                            int numParticles, real *dtBlockShared, int *blockCount) {
        ExecutionPolicy executionPolicy(multiProcessorCount, 256);
        return cuda::launch(true, executionPolicy, ::GodunovNS::Kernel::selectTimestep, simulationTime,
                            particles, numParticles, dtBlockShared, blockCount);
}

__global__ void GodunovNS::Kernel::update(Particles *particles, int numParticles, real dt) {

    real m, vxOld, Px;
#if DIM > 1
    real vyOld, Py;
#if DIM == 3
    real vzOld, Pz;
#endif
#endif
    int i;
    //particle loop
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x){

//        if (abs(particles->massFlux[i]) > ENERGY_FLOOR ||
//            abs(particles->vxFlux[i]) > ENERGY_FLOOR || abs(particles->vyFlux[i]) > ENERGY_FLOOR ||
//            abs(particles->vzFlux[i]) > ENERGY_FLOOR || abs(particles->energyFlux[i]) > ENERGY_FLOOR){
//            printf("Updating particle %i: mF = %e, PxF = %e, PyF = %e, PzF = %e, eF = %e\n", i,
//                   particles->massFlux[i], particles->vxFlux[i], particles->vyFlux[i], particles->vzFlux[i],
//                   particles->energyFlux[i]);
//        }

        m = particles->mass[i];

        // store old velocities for position update
        vxOld = particles->vx[i];
        Px = m*vxOld;
#if DIM > 1
        vyOld = particles->vy[i];
        Py = m*vyOld;
#if DIM == 3
        vzOld = particles->vz[i];
        Pz = m*vzOld;
#endif
#endif
        // set total energy (without gravitational energy)
        particles->u[i] = m*(particles->e[i] + .5*(vxOld*vxOld
#if DIM > 1
                             + vyOld*vyOld
#if DIM == 3
                             + vzOld*vzOld
#endif
#endif
                            ));

#if !MFV_FIX_PARTICLES
        /// update position
        // to stay consistent with the effective face movement vFrame and the appropriately
        // computed fluxes, we're updating the particle positions with a first order Euler step
        particles->x[i] += vxOld*dt;
#if DIM > 1
        particles->y[i] += vyOld*dt;
#if DIM == 3
        particles->z[i] += vzOld*dt;
#endif
#endif
#endif // !MFV_FIX_PARTICLES

        /// update mass
        particles->mass[i] -= dt*particles->massFlux[i];

#if SAFETY_LEVEL
        if (particles->mass[i] < 0. || isnan(particles->mass[i])){
            cudaTerminate("Godunov ERROR: Mass is negative or nan!! m[%i] = %e, massFLux = %e\n", i, particles->mass[i],
                          particles->massFlux[i]);
        }
#endif
        //if(i == 195 || i == 223 || i == 232 || i == 233){
        //    printf("DEBUG: m[%i] = %e, mF = %e, e = %e, u = %e, uF = %e,\n       x = [%e, %e, %e]\n"
        //           , i, m, particles->massFlux[i], particles->e[i], particles->u[i], particles->energyFlux[i],
        //           particles->x[i], particles->y[i], particles->z[i]);
        //}

        /// update velocity
        particles->vx[i] = (Px - dt*particles->vxFlux[i])/particles->mass[i];
#if DIM > 1
        particles->vy[i] = (Py - dt*particles->vyFlux[i])/particles->mass[i];
#if DIM == 3
        particles->vz[i] = (Pz - dt*particles->vzFlux[i])/particles->mass[i];
#endif
#endif
        /// update internal energy
        // update total energy
        particles->u[i] -= dt*particles->energyFlux[i];
        particles->e[i] = particles->u[i]/particles->mass[i]
                -.5*(particles->vx[i]*particles->vx[i]
#if DIM > 1
                     + particles->vy[i]*particles->vy[i]
#if DIM == 3
                     + particles->vz[i]*particles->vz[i]
#endif
#endif
                                              );

        if(particles->e[i] < ENERGY_FLOOR){
            printf("WARNING: Very small or negative internal energy: e[%i] = %e. Applying floor e = %e\n"
                   "         u = %e, energyFlux = %e\n", i, particles->e[i], ENERGY_FLOOR, particles->u[i],
                   particles->energyFlux[i]);
            particles->e[i] = ENERGY_FLOOR;
        }

        /// update position
//        particles->x[i] += .5*(particles->vx[i]+vxOld)*dt;
//#if DIM > 1
//        particles->y[i] += .5*(particles->vy[i]+vyOld)*dt;
//#if DIM == 3
//        particles->z[i] += .5*(particles->vz[i]+vzOld)*dt;
//#endif
//#endif
    }
}

real GodunovNS::Kernel::Launch::update(Particles *particles, int numParticles, real dt) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::GodunovNS::Kernel::update, particles, numParticles, dt);

}
