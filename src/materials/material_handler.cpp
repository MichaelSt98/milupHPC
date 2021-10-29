#include "../../include/materials/material_handler.h"

int LibConfigReader::loadConfigFromFile(char *configFile)
{
    int numberOfElements;

    std::ifstream f(configFile);
    if(f.good()) {
        printf("File exists!\n");
    }
    else {
        fprintf(stderr, "Error: config file %s cannot be found!\n", configFile);
    }
    //config_t config;
    config_init(&config);

    if (!config_read_file(&config, configFile)) {
        fprintf(stderr, "Error reading config file %s.\n", configFile);
        const char *errorText;
        errorText = new char[500];
        errorText = config_error_text(&config);
        int errorLine = config_error_line(&config);
        fprintf(stderr, "Since: %s on %i\n", errorText, errorLine);
        delete [] errorText;
        config_destroy(&config);
        exit(1);
    }

    materials = config_lookup(&config, "materials");
    if (materials != NULL) {
        numberOfElements = config_setting_length(materials);
        int i, j;
        int maxId = 0;
        config_setting_t *material; //, *subset;

        // find max ID of materials
        for (i = 0; i < numberOfElements; ++i) {
            material = config_setting_get_elem(materials, i);
            int ID;
            if (!config_setting_lookup_int(material, "ID", &ID)) {
                fprintf(stderr, "Error. Found material without ID in config file...\n");
                exit(1);
            }

            fprintf(stdout, "Found material ID: %d\n", ID);

            maxId = std::max(ID, maxId);
        }
        if (maxId != numberOfElements - 1) {
            fprintf(stderr, "Error. Material-IDs in config file have to be 0, 1, 2,...\n");
            exit(1);
        }
    }
    return numberOfElements;
}

MaterialHandler::MaterialHandler(integer numMaterials) : numMaterials(numMaterials) {

    h_materials = new Material[numMaterials];
    cuda::malloc(d_materials, numMaterials);

    h_materials[0].ID = 0;
    h_materials[0].interactions = 0;
    //h_materials[0].artificialViscosity.alpha = 3.1;
    h_materials[0].artificialViscosity = ArtificialViscosity();

}

MaterialHandler::MaterialHandler(char *material_cfg) {
    LibConfigReader libConfigReader;
    numMaterials = libConfigReader.loadConfigFromFile(material_cfg);

    config_setting_t *material, *subset;

    h_materials = new Material[numMaterials];
    cuda::malloc(d_materials, numMaterials);

    double temp;

    for (int i = 0; i < numMaterials; ++i) {

        // general
        material = config_setting_get_elem(libConfigReader.materials, i);
        int id;
        config_setting_lookup_int(material, "ID", &id);
        h_materials[id].ID = id;
        fprintf(stdout, "Reading information about material ID %d out of %d in total...\n", id, numMaterials);
        config_setting_lookup_int(material, "interactions", &h_materials[id].interactions);

        // eos
        subset = config_setting_get_member(material, "eos");
        //config_setting_lookup_float(subset, "polytropic_K", &h_materials[id].eos.polytropic_K);
        //config_setting_lookup_float(subset, "polytropic_gamma", &h_materials[id].eos.polytropic_gamma);

        config_setting_lookup_float(subset, "polytropic_K", &temp);
        printf("temp = %f\n", temp);
        h_materials[id].eos.polytropic_K = temp;
        config_setting_lookup_float(subset, "polytropic_gamma", &temp);
        h_materials[id].eos.polytropic_gamma = temp;
    }


}

MaterialHandler::MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta) :
                                    numMaterials(numMaterials) {

    h_materials = new Material[numMaterials];
    cuda::malloc(d_materials, numMaterials);

    h_materials[0].ID = ID;
    h_materials[0].interactions = interactions;
    //h_materials[0].artificialViscosity.alpha = 3.1;
    h_materials[0].artificialViscosity = ArtificialViscosity(alpha, beta);

}

MaterialHandler::~MaterialHandler() {

    delete [] h_materials;
    cuda::free(d_materials);

}

void MaterialHandler::copy(To::Target target, integer index) {

    if (index >= 0 && index < numMaterials) {
        cuda::copy(&h_materials[index], &d_materials[index], 1, target);
    }
    else {
        cuda::copy(h_materials, d_materials, numMaterials, target);
    }

}

void MaterialHandler::communicate(int from, int to, bool fromDevice, bool toDevice) {

    if (fromDevice) { copy(To::host); }

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

    if (toDevice) { copy(To::device); }
}

void MaterialHandler::broadcast(int root, bool fromDevice, bool toDevice) {

    if (fromDevice) { copy(To::host); }

    boost::mpi::environment env;
    boost::mpi::communicator comm;

    boost::mpi::broadcast(comm, h_materials, numMaterials, root);

    if (toDevice) { copy(To::device); }
}