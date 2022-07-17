#include "../../include/materials/material_handler.h"

int LibConfigReader::loadConfigFromFile(const char *configFile)
{
    int numberOfElements;

    try {

        config.readFile(configFile);
        const libconfig::Setting &materials = config.lookup("materials");

        try {

            numberOfElements = materials.getLength();
            int i, j;
            int maxId = 0;

            // find max ID of materials
            for (i = 0; i < numberOfElements; ++i) {

                const libconfig::Setting &material = materials[i];
                int ID;

                if (!(material.lookupValue("ID", ID))) {
                    Logger(ERROR) << "Error. Found material without ID in config file...";
                    MPI_Finalize();
                    exit(1);
                }

                Logger(DEBUG) << "Found material ID: " << ID;
                maxId = std::max(ID, maxId);
            }
            if (maxId != numberOfElements - 1) {
                Logger(ERROR) << "Material-IDs in config file have to be 0, 1, 2,...";
                MPI_Finalize();
                exit(1);
            }
        }
        catch(...) {

        }
    }
    catch(const libconfig::FileIOException &fileIOException) {
        std::cerr << "I/O error while reading file." << std::endl;
        MPI_Finalize();
        exit(1);
    }
    catch(const libconfig::ParseException &parseException) {
        std::cerr << "Parse error at " << parseException.getFile() << ":" << parseException.getLine()
                  << " - " << parseException.getError() << std::endl;
    }
    return numberOfElements;
}

MaterialHandler::MaterialHandler(integer numMaterials) : numMaterials(numMaterials) {

    h_materials = new Material[numMaterials];
#if TARGET_GPU
    cuda::malloc(d_materials, numMaterials);
#endif

    h_materials[0].ID = 0;
    h_materials[0].interactions = 0;
    h_materials[0].artificialViscosity = ArtificialViscosity();

}

MaterialHandler::MaterialHandler(const char *material_cfg) {

    LibConfigReader libConfigReader;
    numMaterials = libConfigReader.loadConfigFromFile(material_cfg);

    h_materials = new Material[numMaterials];
#if TARGET_GPU
    cuda::malloc(d_materials, numMaterials);
#endif

    double temp;

    const libconfig::Setting &materials = libConfigReader.config.lookup("materials");

    for (int i = 0; i < numMaterials; ++i) {

        // general
        const libconfig::Setting &material = materials[i]; //libConfigReader.materials[i];
        int id;

        material.lookupValue("ID", id);
        h_materials[id].ID = id;
        Logger(DEBUG) << "Reading information about material ID " << id << " out of " << numMaterials << "...";
        material.lookupValue("interactions", h_materials[id].interactions);
        material.lookupValue("sml", h_materials[id].sml);

        // artificial viscosity
        const libconfig::Setting &subset_artVisc = material.lookup("artificial_viscosity");
        subset_artVisc.lookupValue("alpha", h_materials[id].artificialViscosity.alpha);
        subset_artVisc.lookupValue("beta", h_materials[id].artificialViscosity.beta);

        // eos
        const libconfig::Setting &subset_eos = material.lookup("eos");
        subset_eos.lookupValue("type", h_materials[id].eos.type);
        subset_eos.lookupValue("polytropic_K", h_materials[id].eos.polytropic_K);
        subset_eos.lookupValue("polytropic_gamma", h_materials[id].eos.polytropic_gamma);
    }


}

MaterialHandler::MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta) :
                                    numMaterials(numMaterials) {

    h_materials = new Material[numMaterials];
#if TARGET_GPU
    cuda::malloc(d_materials, numMaterials);
#endif

    h_materials[0].ID = ID;
    h_materials[0].interactions = interactions;
    //h_materials[0].artificialViscosity.alpha = 3.1;
    h_materials[0].artificialViscosity = ArtificialViscosity(alpha, beta);

}

MaterialHandler::~MaterialHandler() {

    delete [] h_materials;
#if TARGET_GPU
    cuda::free(d_materials);
#endif

}

void MaterialHandler::copy(To::Target target, integer index) {

#if TARGET_GPU
    if (index >= 0 && index < numMaterials) {
        cuda::copy(&h_materials[index], &d_materials[index], 1, target);
    }
    else {
        cuda::copy(h_materials, d_materials, numMaterials, target);
    }
#endif

}

void MaterialHandler::communicate(int from, int to, bool fromDevice, bool toDevice) {

    if (fromDevice) { copy(To::host); }

    boost::mpi::environment env;
    boost::mpi::communicator comm;

    std::vector<boost::mpi::request> reqParticles;
    std::vector<boost::mpi::status> statParticles;


    if (comm.rank() == from) {
        reqParticles.push_back(comm.isend(to, 17, &h_materials[0], numMaterials));
    }
    else {
        statParticles.push_back(comm.recv(from, 17, &h_materials[0], numMaterials));
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

#if TARGET_GPU
    if (toDevice) { copy(To::device); }
#endif

}

void MaterialHandler::broadcast(int root, bool fromDevice, bool toDevice) {

    if (fromDevice) { copy(To::host); }

    boost::mpi::environment env;
    boost::mpi::communicator comm;

    boost::mpi::broadcast(comm, h_materials, numMaterials, root);

#if TARGET_GPU
    if (toDevice) { copy(To::device); }
#endif

}