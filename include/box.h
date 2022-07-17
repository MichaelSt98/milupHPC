#ifndef MILUPHPC_CPP_BOX_H
#define MILUPHPC_CPP_BOX_H

#include "parameter.h"

class Box {

public:

    real minX, maxX;
#if DIM > 1
    real minY, maxY;
#if DIM == 3
    real minZ, maxZ;
#endif
#endif

    Box();

#if DIM == 1
    Box(real minX, real maxX);
#elif DIM == 2
    Box(real minX, real maxX, real minY, real maxY);
#else
    Box(real minX, real maxX, real minY, real maxY, real minZ, real maxZ);
#endif

    const Box& operator=(const Box& box);

    real getCenterX();
#if DIM > 1
    real getCenterY();
#if DIM == 3
    real getCenterZ();
#endif
#endif

};

#endif //MILUPHPC_CPP_BOX_H
