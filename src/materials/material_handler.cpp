#include "../../include/materials/material_handler.h"

MaterialHandler::MaterialHandler(integer numMaterials) : numMaterials(numMaterials) {

    h_materials = new Material[numMaterials];
    gpuErrorcheck(cudaMalloc((void**)&d_materials, numMaterials * sizeof(Material)));

    h_materials[0].ID = 0;
    h_materials[0].interactions = 0;
    //h_materials[0].artificialViscosity.alpha = 3.1;
    h_materials[0].artificialViscosity = ArtificialViscosity();

    //gpuErrorcheck(cudaMemcpy(d_materials, h_materials, numMaterials * sizeof(Material), cudaMemcpyHostToDevice));

}

MaterialHandler::MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta) :
                                    numMaterials(numMaterials) {

    h_materials = new Material[numMaterials];
    gpuErrorcheck(cudaMalloc((void**)&d_materials, numMaterials * sizeof(Material)));

    h_materials[0].ID = ID;
    h_materials[0].interactions = interactions;
    //h_materials[0].artificialViscosity.alpha = 3.1;
    h_materials[0].artificialViscosity = ArtificialViscosity(alpha, beta);

    //gpuErrorcheck(cudaMemcpy(d_materials, h_materials, numMaterials * sizeof(Material), cudaMemcpyHostToDevice));

}

MaterialHandler::~MaterialHandler() {

    delete [] h_materials;
    gpuErrorcheck(cudaFree(d_materials));

}

void MaterialHandler::toDevice(integer index) {
    if (index >= 0 && index < numMaterials) {
        gpuErrorcheck(cudaMemcpy(&d_materials[index], &h_materials[index], sizeof(Material),
                                 cudaMemcpyHostToDevice));
    }
    else {
        gpuErrorcheck(cudaMemcpy(d_materials, h_materials, numMaterials * sizeof(Material),
                                 cudaMemcpyHostToDevice));
    }
}

void MaterialHandler::toHost(integer index) {
    if (index >= 0 && index < numMaterials) {
        gpuErrorcheck(cudaMemcpy(&h_materials[index], &d_materials[index], sizeof(Material),
                                 cudaMemcpyDeviceToHost));
    }
    else {
        gpuErrorcheck(cudaMemcpy(&h_materials[index], &d_materials[index], numMaterials * sizeof(Material),
                                 cudaMemcpyDeviceToHost));
    }
}

void MaterialHandler::communicate(int from, int to, bool fromDevice, bool toDevice) {

    if (fromDevice) { this->toHost(); }

    boost::mpi::environment env;
    boost::mpi::communicator comm;

    //printf("numMaterials = %i    comm.rank() = %i\n", numMaterials, comm.rank());

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;


    if (comm.rank() == from) {
        reqParticles.push_back(comm.isend(to, 17, &h_materials[0], numMaterials));
    }
    else {
        statParticles.push_back(comm.recv(from, 17, &h_materials[0], numMaterials));
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    if (toDevice) { this->toDevice(); }
}

void MaterialHandler::broadcast(int root, bool fromDevice, bool toDevice) {

    if (fromDevice) { this->toHost(); }

    boost::mpi::environment env;
    boost::mpi::communicator comm;

    boost::mpi::broadcast(comm, h_materials, numMaterials, root);

    if (toDevice) { this->toDevice(); }
}