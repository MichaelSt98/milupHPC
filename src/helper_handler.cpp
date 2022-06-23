#include "../include/helper_handler.h"

HelperHandler::HelperHandler(int numProcesses, int numParticlesLocal, int numParticles, int sumParticles, int numNodes) :
            numProcesses(numProcesses), numParticlesLocal(numParticlesLocal), numParticles(numParticles),
            sumParticles(sumParticles), numNodes(numNodes) {

    cuda::malloc(d_integerVal, 1);
    cuda::malloc(d_integerVal1, 1);
    cuda::malloc(d_integerVal2, 1);

    cuda::malloc(d_realVal, 1);
    cuda::malloc(d_realVal1, 1);
    cuda::malloc(d_realVal2, 1);

    cuda::malloc(d_keyTypeVal, 1);

    cuda::malloc(d_integerBuffer, numNodes);
    cuda::malloc(d_integerBuffer1, numNodes);
    cuda::malloc(d_integerBuffer2, numNodes);
    cuda::malloc(d_integerBuffer3, numNodes);
    cuda::malloc(d_integerBuffer4, numNodes);

    cuda::malloc(d_sendCount, numProcesses); // + 1
    cuda::malloc(d_sendCount1, numProcesses); // + 1

    cuda::malloc(d_idIntegerBuffer, numParticles);
    cuda::malloc(d_idIntegerBuffer1, numParticles);

    cuda::malloc(d_realBuffer, numNodes);
    cuda::malloc(d_realBuffer1, numNodes);

    cuda::malloc(d_keyTypeBuffer, numParticles);
    cuda::malloc(d_keyTypeBuffer1, sumParticles);
    cuda::malloc(d_keyTypeBuffer2, sumParticles);

    cuda::malloc(d_helper, 1);

    //HelperNS::Kernel::Launch::set(d_helper, d_integerVal, d_realVal, d_keyTypeVal, d_integerBuffer,
                                  //d_realBuffer, d_keyTypeBuffer);

    HelperNS::Kernel::Launch::set(d_helper, d_integerVal, d_integerVal1, d_integerVal2,
                                  d_realVal, d_realVal1, d_realVal2, d_keyTypeVal,
                                  d_integerBuffer, d_integerBuffer1, d_integerBuffer2,
                                  d_integerBuffer3, d_integerBuffer4,
                                  d_sendCount, d_sendCount1, d_idIntegerBuffer,
                                  d_idIntegerBuffer1, d_realBuffer, d_realBuffer1,
                                  d_keyTypeBuffer, d_keyTypeBuffer1, d_keyTypeBuffer2);

}

HelperHandler::~HelperHandler() {

    cuda::free(d_integerVal);
    cuda::free(d_integerVal1);
    cuda::free(d_integerVal2);

    cuda::free(d_realVal);
    cuda::free(d_realVal1);
    cuda::free(d_realVal2);

    cuda::free(d_keyTypeVal);

    cuda::free(d_integerBuffer);
    cuda::free(d_integerBuffer1);
    cuda::free(d_integerBuffer2);
    cuda::free(d_integerBuffer3);
    cuda::free(d_integerBuffer4);

    cuda::free(d_sendCount);
    cuda::free(d_sendCount1);

    cuda::free(d_idIntegerBuffer);
    cuda::free(d_idIntegerBuffer1);

    cuda::free(d_realBuffer);
    cuda::free(d_realBuffer1);

    cuda::free(d_keyTypeBuffer);
    cuda::free(d_keyTypeBuffer1);
    cuda::free(d_keyTypeBuffer2);

    cuda::free(d_helper);

}

void HelperHandler::reset() {

    cuda::set(d_integerVal, 0, 1);
    cuda::set(d_integerVal1, 0, 1);
    cuda::set(d_integerVal2, 0, 1);

    cuda::set(d_realVal, (real)0, 1);
    cuda::set(d_realVal1, (real)0, 1);
    cuda::set(d_realVal2, (real)0, 1);

    cuda::set(d_keyTypeVal, (keyType)0, 1);

    cuda::set(d_integerBuffer, 0, numNodes);
    cuda::set(d_integerBuffer1, 0, numParticles);
    cuda::set(d_integerBuffer2, 0, numParticles);
    cuda::set(d_integerBuffer3, 0, numParticles);
    cuda::set(d_integerBuffer4, 0, numParticles);

    cuda::set(d_sendCount, 0, numProcesses);
    cuda::set(d_sendCount1, 0, numProcesses);

    cuda::set(d_idIntegerBuffer, (idInteger)0, numParticles);
    cuda::set(d_idIntegerBuffer1, (idInteger)0, numParticles);

    cuda::set(d_realBuffer, (real)0, numNodes);
    cuda::set(d_realBuffer1, (real)0, numNodes);

    cuda::set(d_keyTypeBuffer, (keyType)0, numParticlesLocal);
    cuda::set(d_keyTypeBuffer1, (keyType)0, sumParticles);
    cuda::set(d_keyTypeBuffer2, (keyType)0, sumParticles);

    //cuda::set(d_integerBuffer, 0, length);
    //cuda::set(d_realBuffer, (real)0, length);
    //cuda::set(d_keyTypeBuffer, (keyType)0, length);
}