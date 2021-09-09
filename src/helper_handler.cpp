#include "../include/helper_handler.h"

HelperHandler::HelperHandler(integer length) : length(length) {

    cuda::malloc(d_integerVal, 1);
    cuda::malloc(d_realVal, 1);
    cuda::malloc(d_keyTypeVal, 1);

    cuda::malloc(d_integerBuffer, length);
    cuda::malloc(d_realBuffer, length);
    cuda::malloc(d_keyTypeBuffer, length);

    cuda::malloc(d_helper, 1);
    HelperNS::Kernel::Launch::set(d_helper, d_integerVal, d_realVal, d_keyTypeVal, d_integerBuffer,
                                  d_realBuffer, d_keyTypeBuffer);

}

HelperHandler::~HelperHandler() {

    cuda::free(d_integerVal);
    cuda::free(d_realVal);
    cuda::free(d_keyTypeVal);

    cuda::free(d_integerBuffer);
    cuda::free(d_realBuffer);
    cuda::free(d_keyTypeBuffer);

    cuda::free(d_helper);

}

void HelperHandler::reset() {
    cuda::set(d_integerVal, 0, 1);
    cuda::set(d_realVal, (real)0, 1);
    cuda::set(d_keyTypeVal, (keyType)0, 1);

    cuda::set(d_integerBuffer, 0, length);
    cuda::set(d_realBuffer, (real)0, length);
    cuda::set(d_keyTypeBuffer, (keyType)0, length);
}