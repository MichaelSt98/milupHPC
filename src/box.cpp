#include "../include/box.h"

Box::Box() {

}

#if DIM == 1
Box::Box(real minX, real maxX) : minX{ minX },
                    maxX{ maxX } {

}
#elif DIM == 2
Box::Box(real minX, real maxX, real minY, real maxY) : minX{ minX },
                    maxX{ maxX }, minY{ minY }, maxY{ maxY } {

}
#else
Box::Box(real minX, real maxX, real minY, real maxY, real minZ, real maxZ) : minX{ minX },
                    maxX{ maxX }, minY{ minY }, maxY{ maxY }, minZ{ minZ }, maxZ{ maxZ} {

}
#endif

const Box& Box::operator=(const Box& box) {

    /*
    minX = box.minX;
    maxX = box.maxX;
#if DIM > 1
    minY = box.minY;
    maxY = box.maxY;
#if DIM == 3
    minZ = box.minZ;
    maxZ = box.maxZ;
#endif
#endif
     */

}


real Box::getCenterX() {
    return maxX - minX;
}

#if DIM > 1

real Box::getCenterY() {
    return maxY - minY;
}

#if DIM == 3

real Box::getCenterZ() {
    return maxZ - minZ;
}

#endif
#endif