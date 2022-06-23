#ifndef NBODY_H5RENDERER_H
#define NBODY_H5RENDERER_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <highfive/H5File.hpp>
#include <boost/filesystem.hpp>
#include <omp.h>

#include <fstream>

#include "Logger.h"
#include "Particle.h"

namespace fs = boost::filesystem;

struct ColorRGB {
    char r;
    char g;
    char b;

    //ColorRGB() : r { 0 }, g { 0 }, b { 0 }{}; // black
    ColorRGB() : r { ~0 }, g { ~0 }, b { ~0 }{};// white
    ColorRGB(char _r, char _g, char _b) : r { _r }, g { _g }, b { _b }{};
};

const ColorRGB COLORS[16] = {
        ColorRGB(120, 23, 16), // dark red
        ColorRGB(7, 14, 82), // dark blue
        ColorRGB(4, 71, 20), // dark green
        ColorRGB(4, 69, 71), // dark turquoise/blue-green
        ColorRGB(116, 120, 13), // "dark" yellow
        ColorRGB(0, 0, 0), // black
        ColorRGB(176, 83,37), // orange
        ColorRGB(48, 48, 48), // (dark) greymake
        ColorRGB(13, 67, 120), // light(er) blue
        ColorRGB(43, 120, 13), // light(er) green
        ColorRGB(207, 53, 23), // light(er) red
        ColorRGB(76, 49, 140), // lila
        ColorRGB(~0, 0, ~0), // magenta
        ColorRGB(48, 33, 27), // brown
        ColorRGB(145, 145, 145), // light(er) grey
        ColorRGB(14, 11, 31) // dark blue/lila
        //ColorRGB(0, ~0, 0), // green
        //ColorRGB(0, 0, ~0), // blue
        //ColorRGB(~0, 0, 127), // pink
        //ColorRGB(~0, ~0, 0), // yellow
        //ColorRGB(~0, 0, ~0), // magenta
        //ColorRGB(~0, 0, 0), // red
        //ColorRGB(0, ~0, ~0), // turquoise
        //ColorRGB(~0, 127, 0), //orange
        //ColorRGB(127, 127, 127), // grey
        //ColorRGB(~0, ~0, ~0) //white
};

class H5Renderer {

private:
    // initializing variables for H5Renderer
    // input variables
    const std::string h5folder;
    const double systemSize;
    const int imgHeight;
    const double zoom;
    const bool processColoring; // default: true
    const bool markParticles;

    // constants
    const double SCALE2FIT = .95;

    // internal variables
    std::vector<fs::path> h5files;
    long psSize; // pixelSpace size

    //functions
    ColorRGB procColor(unsigned long k, const std::vector<unsigned long> &ranges);
    int procNumber(unsigned long k, const std::vector<unsigned long> &ranges);
    void clearPixelSpace(ColorRGB *pixelSpace);
    int pos2pixel(double pos);
    void particle2PixelXY(double x, double y, const ColorRGB &color, ColorRGB *pixelSpace);
    void particle2PixelXZ(double x, double z, const ColorRGB &color, ColorRGB *pixelSpace);
    void pixelSpace2File(const std::string &outFile, ColorRGB *pixelSpace);

public:
    H5Renderer(std::string _h5folder, double _systemSize, int _imgHeight, double _zoom, bool _markParticles,
               bool _processColoring=true);

    void createImages(std::string outDir);
};


#endif //NBODY_H5RENDERER_H
