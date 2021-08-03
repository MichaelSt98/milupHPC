//
// Created by Johannes Martin on 31.05.21.
//

#ifndef H5RENDERER_PARTICLE_H
#define H5RENDERER_PARTICLE_H

class Particle {

public:
    double x;
    double y;
    double z;

    unsigned long key;

    Particle (double _x, double _y, double _z, unsigned long _key) : x { _x }, y { _y }, z { _z }, key { _key }{}

    // comparator functions
    static bool zComp(Particle p1, Particle p2){
        return p1.z < p2.z;
    }
    static bool yComp(Particle p1, Particle p2){
        return p1.y < p2.y;
    }

};

#endif //H5RENDERER_PARTICLE_H
