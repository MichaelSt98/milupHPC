#include "../../include/integrator/base_integrator.cuh"


BaseIntegrator::BaseIntegrator() {
    printf("BaseIntegrator()\n");
}

BaseIntegrator::~BaseIntegrator() {
    printf("~BaseIntegrator()\n");
}

void BaseIntegrator::rhs() {
    printf("BaseIntegrator::rhs()\n");
}