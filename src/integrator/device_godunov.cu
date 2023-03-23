#include "../../include/integrator/device_godunov.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

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
        // to stay consistent with the effective face movement v_Frame and the appropriately
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
