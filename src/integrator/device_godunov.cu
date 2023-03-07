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

        /// update mass
        particles->mass[i] -= dt*particles->massFlux[i];

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

        /// update position
        particles->x[i] += .5*(particles->vx[i]+vxOld)*dt;
#if DIM > 1
        particles->y[i] += .5*(particles->vy[i]+vxOld)*dt;
#if DIM == 3
        particles->z[i] += .5*(particles->vz[i]+vxOld)*dt;
#endif
#endif

    }
}

real GodunovNS::Kernel::Launch::update(Particles *particles, int numParticles, real dt) {
    ExecutionPolicy executionPolicy;
    return cuda::launch(true, executionPolicy, ::GodunovNS::Kernel::update, particles, numParticles, dt);

}
