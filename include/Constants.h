#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#define KEY_MAX ULONG_MAX
#define DOMAIN_LIST_SIZE 512

typedef struct SimulationParameters
{

    //bool debug;
    //bool benchmark;
    //bool fullscreen;
    bool timeKernels;
    int iterations;
    int numberOfParticles;
    float timestep;
    float gravity;
    float dampening;
    int gridSize;
    int blockSize;
    int warp;
    int stackSize;
    int renderInterval;
    bool loadBalancing;
    int loadBalancingInterval;
    int curveType;

} SimulationParameters;

/// Physical constants
const double PI = 3.14159265358979323846;   //! Pi
const double TO_METERS = 1.496e11;          //! AU to meters
const double G = 6.67408e-11;               //! Gravitational constant


/// Rendering related
const int WIDTH = 512;
const int HEIGHT = 512;
const int DEPTH = 512;
const double RENDER_SCALE = 1.2;
const double MAX_VEL_COLOR = 1;
const double MIN_VEL_COLOR = 0.0001;
const double PARTICLE_BRIGHTNESS = 0.35;
const double PARTICLE_SHARPNESS = 1.0;
const int DOT_SIZE = 8;
const int RENDER_INTERVAL = 2; // How many timesteps to simulate in between each frame rendered


#endif /* CONSTANTS_H_ */
