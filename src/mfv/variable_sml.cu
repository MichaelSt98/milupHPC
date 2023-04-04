#include "../../include/mfv/variable_sml.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"

namespace MFV {

    namespace Kernel{

        __global__ void guessSML(Particles *particles, int numParticles, real dt){
            for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numParticles; i+= blockDim.x * gridDim.x){
                /// set initial guess for iteration scheme to find an appropriate kernel size
                particles->sml[i] += dt * particles->sml[i] * particles->vxGrad[i] / DIM;
#if DIM > 1
                particles->sml[i] += dt * particles->sml[i] * particles->vyGrad[i+1] / DIM;
#if DIM ==3
                particles->sml[i] += dt * particles->sml[i] * particles->vzGrad[i+2] / DIM;
#endif
#endif
            }

        }


        __global__ void variableSML_FRNN(Tree *tree, Particles *particles, integer *interactions, real radius,
                                         integer numParticlesLocal, integer numParticles, integer numNodes,
                                         Material *materials, ::SPH::SPH_kernel kernel){

            register int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            register int stride = blockDim.x * gridDim.x;
            register int offset = 0;
            register int index;

            register integer childNumber, nodeIndex, depth, childIndex;

            register real dx, x;
#if DIM > 1
            register real dy, y;
#if DIM == 3
            register real dz, z;
#endif
#endif

            register real d, r, interactionDistance;

            register int noOfInteractions;

            register int currentNodeIndex[MAX_DEPTH];
            register int currentChildNumber[MAX_DEPTH];

            real sml, smlj;

            // variable smoothing length for MFV/MFM additions
            bool foundSML;
            real omg, NNN, targetNNN, dndh;
            int smlIt = 0;
#if DIM == 1
            constexpr real C_d = 1.;
#elif DIM == 2
            constexpr real C_d = M_PI;
#else // DIM == 3
            constexpr real C_d = 4.*M_PI/3.;
#endif

            while ((bodyIndex + offset) < numParticlesLocal) {

                //index = tree->sorted[bodyIndex + offset];
                index = bodyIndex + offset;

                x = particles->x[index];
#if DIM > 1
                y = particles->y[index];
#if DIM == 3
                z = particles->z[index];
#endif
#endif
                targetNNN = (real)materials[particles->materialId[index]].interactions; // effective number of neighbors
                foundSML = false;

                while(!foundSML) { // repeat actual neighbor search until appropriate sml has been found
                    sml = particles->sml[index];

                    // resetting
                    //for (integer i = 0; i < MAX_NUM_INTERACTIONS; i++) {
                    //    interactions[(bodyIndex + offset) * MAX_NUM_INTERACTIONS + i] = -1;
                    //}
                    //numberOfInteractions[bodyIndex + offset] = 0;
                    // end: resetting

                    depth = 0;
                    currentNodeIndex[depth] = 0; //numNodes - 1;
                    currentChildNumber[depth] = 0;
                    noOfInteractions = 0;

                    r = radius * 0.5;

                    interactionDistance = (r + sml);

                    do {

                        childNumber = currentChildNumber[depth];
                        nodeIndex = currentNodeIndex[depth];

                        while (childNumber < POW_DIM) {

                            childIndex = tree->child[POW_DIM * nodeIndex + childNumber];
                            childNumber++;

                            if (childIndex != -1 && childIndex != index) {

                                smlj = particles->sml[childIndex];

                                dx = x - particles->x[childIndex]; //x_child;
#if DIM > 1
                                dy = y - particles->y[childIndex]; //y_child;
#if DIM == 3
                                dz = z - particles->z[childIndex]; //z_child;
#endif
#endif
                                // its a leaf
                                if (childIndex < numParticles) {
#if DIM == 1
                                    d = dx*dx;
#elif DIM == 2
                                    d = dx*dx + dy*dy;
#else
                                    d = dx*dx + dy*dy + dz*dz;
#endif

                                    //if ((bodyIndex + offset) % 1000 == 0) {
                                    //printf("sph: index = %i, d = %i\n", bodyIndex+offset, d);
                                    //}

                                    if (d < (sml * sml) &&
                                        d < (smlj * smlj)) {
                                        interactions[index * MAX_NUM_INTERACTIONS + noOfInteractions] = childIndex;
                                        // debug
                                        //if (noOfInteractions > MAX_NUM_INTERACTIONS) {
                                        //    printf("%i: noOfInteractions = %i > MAX_NUM_INTERACTIONS = %i (sml = %e) (%e, %e, %e)\n",
                                        //           bodyIndex + offset, noOfInteractions, MAX_NUM_INTERACTIONS, particles->sml[bodyIndex + offset],
                                        //           particles->x[bodyIndex + offset], particles->y[bodyIndex + offset],
                                        //           particles->y[bodyIndex + offset]);
                                        //    assert(0);
                                        //}
                                        // end: debug
                                        // debug: should not happen
                                        //if ((bodyIndex + offset) == childIndex) {
                                        //    printf("fixedRadiusNN: index = %i == childIndex = %i\n",
                                        //           bodyIndex + offset, childIndex);
                                        //    assert(0);
                                        //}
                                        // end: debug
                                        if (noOfInteractions > MAX_NUM_INTERACTIONS) {
                                            cudaTerminate("noOfInteractions = %i > MAX_NUM_INTERACTIONS = %i\n",
                                                          noOfInteractions, MAX_NUM_INTERACTIONS);
                                        }
                                        noOfInteractions++;
                                    }
                                }
#if DIM == 1
                                    else if (cuda::math::abs(dx) < interactionDistance) {
#elif DIM == 2
                                    else if (cuda::math::abs(dx) < interactionDistance &&
                                     cuda::math::abs(dy) < interactionDistance) {
#else
                                else if ((cuda::math::abs(dx) < interactionDistance &&
                                          cuda::math::abs(dy) < interactionDistance &&
                                          cuda::math::abs(dz) < interactionDistance) || particles->nodeType[childIndex] >= 1) {
#endif

                                    currentChildNumber[depth] = childNumber;
                                    currentNodeIndex[depth] = nodeIndex;

                                    depth++;
                                    r *= 0.5;

                                    interactionDistance = (r + sml);
                                    if (depth > MAX_DEPTH) { //MAX_DEPTH) {
                                        // TODO: why not here redoNeighborSearch() ???
                                        cudaTerminate("depth = %i > MAX_DEPTH = %i\n", depth, MAX_DEPTH);
                                    }
                                    childNumber = 0;
                                    nodeIndex = childIndex;
                                }
                            }
                        }

                        depth--;
                        r *= 2.0;
                        interactionDistance = (r + sml);

                    } while (depth >= 0);

                    //printf("First neighbor search done for particle %i, noi = %i, targetNNN = %e\n",
                    //       index, noOfInteractions, targetNNN);

                    particles->noi[index] = noOfInteractions;

                    dndh = ::MFV::Compute::inverseVolume(omg, index, kernel, particles, interactions);

                    NNN = C_d*pow(sml,DIM)*omg;

                    if (targetNNN-ROOT_FOUND_TOL_SML < NNN
                        && targetNNN+ROOT_FOUND_TOL_SML > NNN) {
                        foundSML = true;
                    } else if (smlIt > MAX_NUM_SML_ITERATIONS){
                        printf("WARNING: No good kernel size found after %i iterations for particle %i. sml = %e, NNN = %e, targetNNN = %e\n",
                               MAX_NUM_SML_ITERATIONS, index, sml, NNN, targetNNN);
                        foundSML = true;
                    } else { // Newton-Raphson iteration
                        particles->sml[index] -= (sml - pow(targetNNN/(C_d*omg), 1./DIM))
                                                 / (1.-1./DIM*pow(targetNNN/(C_d*omg), 1./DIM-1.)
                                                 *pow(omg, -2.)*dndh);
                        //printf("Reiterate for particle %i with omg = %e, dndh = %e, old sml = %e, NNN = %e and new sml = %e\n",
                        //       index, omg, dndh, sml, NNN, particles->sml[index]);
                        smlIt++;
                    }

                }

                particles->omega[index] = omg;
                particles->rho[index] = particles->mass[index]*omg; // update density

                offset += stride;
                __syncthreads(); // TODO: why?
            }
        }

        namespace Launch{
            real variableSML_FRNN(Tree *tree, Particles *particles, integer *interactions, real radius,
                                  integer numParticlesLocal, integer numParticles, integer numNodes,
                                  Material *materials, ::SPH::SPH_kernel kernel) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::MFV::Kernel::variableSML_FRNN, tree, particles, interactions,
                                    radius, numParticlesLocal, numParticles, numNodes, materials, kernel);
            }

            real guessSML(Particles *particles, int numParticles, real dt) {
                ExecutionPolicy executionPolicy;
                return cuda::launch(true, executionPolicy, ::MFV::Kernel::guessSML, particles, numParticles, dt);
            }
        }
    }
}