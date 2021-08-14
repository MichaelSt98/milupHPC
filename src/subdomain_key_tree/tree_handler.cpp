//
// Created by Michael Staneker on 14.08.21.
//

#include "../../include/subdomain_key_tree/tree_handler.h"

TreeHandler::TreeHandler(integer numParticles, integer numNodes) {

    gpuErrorcheck(cudaMalloc((void**)&d_count, numNodes * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_start, numNodes * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_sorted, numNodes * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_child, POW_DIM * numNodes * sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_index, sizeof(integer)));
    gpuErrorcheck(cudaMalloc((void**)&d_mutex, sizeof(integer)));

    gpuErrorcheck(cudaMalloc((void**)&d_minX, sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_maxX, sizeof(real)));
    h_minX = new real;
    h_maxX = new real;
#if DIM > 1
    gpuErrorcheck(cudaMalloc((void**)&d_minY, sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_maxY, sizeof(real)));
    h_minY = new real;
    h_maxY = new real;
#if DIM == 3
    gpuErrorcheck(cudaMalloc((void**)&d_minZ, sizeof(real)));
    gpuErrorcheck(cudaMalloc((void**)&d_maxZ, sizeof(real)));
    h_minZ = new real;
    h_maxZ = new real;
#endif
#endif

    gpuErrorcheck(cudaMalloc((void**)&d_tree, sizeof(Tree)));

#if DIM == 1
    TreeNS::launchSetKernel(d_tree, d_count, d_start, d_child, d_sorted, d_index, d_minX, d_maxX);
#elif DIM == 2
    TreeNS::launchSetKernel(d_tree, d_count, d_start, d_child, d_sorted, d_index, d_minX, d_maxX, d_minY, d_maxY);
#else
    TreeNS::launchSetKernel(d_tree, d_count, d_start, d_child, d_sorted, d_index, d_minX, d_maxX, d_minY, d_maxY,
                            d_minZ, d_maxZ);
#endif

}

TreeHandler::~TreeHandler() {

    gpuErrorcheck(cudaFree(d_count));
    gpuErrorcheck(cudaFree(d_start));
    gpuErrorcheck(cudaFree(d_sorted));
    gpuErrorcheck(cudaFree(d_child));
    gpuErrorcheck(cudaFree(d_index));
    gpuErrorcheck(cudaFree(d_mutex));

    gpuErrorcheck(cudaFree(d_minX));
    gpuErrorcheck(cudaFree(d_maxX));
    delete h_minX;
    delete h_maxX;
#if DIM > 1
    gpuErrorcheck(cudaFree(d_minY));
    gpuErrorcheck(cudaFree(d_maxY));
    delete h_minY;
    delete h_maxY;
#if DIM == 3
    gpuErrorcheck(cudaFree(d_minZ));
    gpuErrorcheck(cudaFree(d_maxZ));
    delete h_minZ;
    delete h_maxZ;
#endif
#endif

}
