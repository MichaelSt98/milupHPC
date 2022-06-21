$\frac{dv}{dt} = \frac{1}{\rho} \vec{\nabla} \vec{\sigma}$

$\frac{d\rho}{dt} = -\rho \nabla \vec{v}$

* not necessary anymore:
	* `#INTEGRATE_ENERGY`: always
	* `#ARTIFICIAL_VISCOSITY`: always
	* `#AVERAGE_KERNELS`: always
	* ..
* `#SELF_GRAVITY`
	* if not used, use tree for SPH (NNS) 
* `#HYDRO`
	* simplest case: density not integrated, but calculated from masses
* `#INTEGRATED_DENSITY` for hydro and solid
	* $\frac{d\rho}{dt} = -\rho \nabla \vec{v}$
		* **calculate** or save?	 	
* `#SOLID`
* `#POROSITY` for hydro and solid
	* models: stress p-alpha, psilon p-alpha, 
* `#PLASTICITY`
	* do not distinguish between different models (via compiler directives) 

## Settings, ...

```cpp
#ifdef SINGLE_PRECISION
    typedef float real;
#else
    typedef double real;
#endif
typedef int integer;
typedef unsigned long keyType;
typedef int idInteger;
```

```cpp
#define MAX_LEVEL 21
#define DEBUGGING 0
/**
 * * `SAFETY_LEVEL 0`: almost no safety measures
 * * `SAFETY_LEVEL 1`: most relevant/important safety measures
 * * `SAFETY_LEVEL 2`: more safety measures, including assertions
 * * `SAFETY_LEVEL 3`: many security measures, including all assertions
 */
#define SAFETY_LEVEL 2
#define DIM 3 // dimension of the problem
#define power_two(x) (1 << (x))
#define POW_DIM power_two(DIM)
#define SI_UNITS 1 // [0]: natural units, [1]: SI units
#define CUBIC_DOMAINS 1 // [0]: rectangular (and not necessarily cubic domains), [1]: cubic domains
#define GRAVITY_SIM 1 // Simulation with gravitational forces
#define SPH_SIM 1 // // SPH simulation
#define INTEGRATE_ENERGY 0 // integrate energy equation
#define INTEGRATE_DENSITY 1 // integrate density equation
#define INTEGRATE_SML 0 // integrate smoothing length
#define VARIABLE_SML 1 // variable smoothing length
...
// * **SPH_EQU_VERSION 1:** original version with
// * **SPH_EQU_VERSION 2:** slighty different version with
#define SPH_EQU_VERSION 1
```

## Important classes (attributes)

Particle class:

```cpp
real *mass; // mass
real *x; // position
real *vx; // velocity
real *ax; // acceleration
real *g_ax; // gravitational acceleration
...
int *nodeType; // particle, pseudo-particle, (lowest) domain list
int *level;
idInteger *uid; // unique identifier (unsigned int/long?)
int *materialId; // material identfier (e.g.: ice, basalt, ...)
real *sml; // smoothing length
int *nnl; // near neighbor list
int *noi; // number of interactions 
real *e; // internal energy
real *dedt; // time derivative of the internal energy
real *cs; // soundspeed
real *rho; // density
real *p; // pressure
real *muijmax; // needed for artificial viscosity
...
```

Tree class:

```cpp
int *child; // children/child nodes or leaves 
int *sorted; // sorted (indices) for better cache performance
int *toDeleteLeaf; // index for rebuilding tree
int *toDeleteNode; // old index for rebuilding tree
real *minX; // bounding box minimal x
real *maxX; // bounding box maximum x
...
```

SubDomainKeyTree class:

```cpp
int rank; // mpi rank
int numProcesses; // mpi comm size
keyType *range; // sfc ranges, mapping key ranges/borders to MPI processes
int *procParticleCounter; // particle counter in dependence of MPI process(es)
```

DomainList:

```cpp
int *domainListIndices; // domain list node indices
int *domainListLevels; // domain list node levels
int *domainListIndex; // domain list node index, thus amount of domain list nodes
int *domainListCounter; // domain list node counter, usable as buffer
keyType *domainListKeys; // domain list node keys
keyType *sortedDomainListKeys; // sorted domain list node keys for sorting the keys
int *relevantDomainListIndices; // reduce/collect info in respect to some criterion
int *relevantDomainListLevels;
int *relevantDomainListProcess;
```

Materials:

```cpp
struct ArtificialViscosity {
  real alpha;
  real beta;
};

struct EqOfSt {
  int type;
  real polytropic_K;
  real polytropic_gamma;
};

class Material {
public:
  int ID; // unique identifier
  int interactions; // max number of interactions (SPH)
  real sml; // smoothing length
  ArtificialViscosity artificialViscosity; 
  EqOfSt eos;
};
```

SimulationTime:

```cpp
real *dt; // time step
real *startTime; // start time
real *subEndTime; // end time for sub-integration-step
real *endTime; // simulation end time
real *currentTime; // current simulation time
real *dt_max; // max time step
...
```

	
## Kernel skeleton

```cpp

//*.cuh
namespace Kernel {
  __global__ void skeleton(Class *class, int n);

  namespace Launch {
    void skeleton(Class *class, int n);
  }
}

//*.cu
__global__ void Kernel::skeleton(Class *class, int n) {

  int bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int offset = 0;
  
    while ((bodyIndex + offset) < n) {
      class->x[bodyIndex + offset] = 0;
      ...    
    	 offset += stride;
    }
}

void Kernel::Launch::skeleton(Class *class, int n) {
  ExecutionPolicy executionPolicy();
  cuda::launch(false, executionPolicy, ::Kernel::skeleton, class, n);
}

//*.cpp

Kernel::Launch::skeleton(class, n);

```

## CUDA

```cpp

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true); {
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
    }
}

class ExecutionPolicy {
public:
  dim3 gridSize;
  dim3 blockSize;
  size_t sharedMemBytes;
  ...
};

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

namespace cuda {

  real launch(bool timeKernel, const ExecutionPolicy &policy, void (*f)(Arguments...), Arguments... args) {
    f<<<p.gridSize, p.blockSize, p.sharedMemBytes>>>(args...); 
  }
                    
  template <typename T>
  void copy(T *h_var, T *d_var, std::size_t count = 1, To::Target copyTo = To::device) {
    switch (copyTo) {
      case To::device: {
        gpuErrorcheck(cudaMemcpy(d_var, h_var, count * sizeof(T), cudaMemcpyHostToDevice));
      } break;
      case To::host: {
        gpuErrorcheck(cudaMemcpy(h_var, d_var, count * sizeof(T), cudaMemcpyDeviceToHost));
      } break;
      default: {
        printf("cuda::copy Target not available!\n");
      }
    }
  }

  template <typename T>
  void set(T *d_var, T val, std::size_t count = 1) {
    gpuErrorcheck(cudaMemset(d_var, val, count * sizeof(T)));
  }

  template <typename T>
  void malloc(T *&d_var, std::size_t count) {
    gpuErrorcheck(cudaMalloc((void**)&d_var, count * sizeof(T)));
  }

  template <typename T>
  void free(T *d_var) {
    gpuErrorcheck(cudaFree(d_var));
  }
  
  namespace math {
    __device__ real min(real a, real b) {
#if SINGLE_PRECISION
      return fminf(a, b);
#else
      return fmin(a, b);
#endif
    }
    __device__ real sqrt(real a) {
#if SINGLE_PRECISION
      return sqrtf(a);
#else
      return ::sqrt(a);
#endif
    }
    ...
  }
}
```

## Handler

```cpp
// foo.cuh/cu
class Foo {

public:
  int *a;
  real *b;
  ...
  
  CUDA_CALLABLE_MEMBER Foo() {}
  CUDA_CALLABLE_MEMBER Foo(int *a, real *b) : a{a}, b{b} {}
  CUDA_CALLABLE_MEMBER void set(int *a, real *b) {
    this->a = a;
    this->b = b;
  }
  CUDA_CALLABLE_MEMBER ~Foo() {}
  ...
};

namespace FooNS {
  namespace Kernel {
  __global__ void set(Foo *foo, int *a, real *b) {
    foo->set(foo, a, b);
  }
    namespace Launch {
      void set(Foo *foo, int *a, real *b) {
        ExecutionPolicy executionPolicy(1, 1);
        cuda::launch(false, executionPolicy, ::FooNS::, foo, a, b);
      }
    }
  }
  ...
}

// foo_handler.cuh/cu
class FooHandler {

public:
  int *h_a;
  real *h_b;
  Foo *h_foo;
  
  int *d_a;
  real *d_b;
  Foo *d_foo;
  
  FooHandler(int n) {
    h_a = new int[n];
    h_b = new real[n];
    
    cuda::malloc(d_a, n);
    cuda::malloc(d_b, n);
    
    h_foo = new Foo();
    h_foo->set(h_a, h_b);
    cuda::malloc(d_foo, 1);
    FooNS::Kernel::Launch(d_foo, d_a, d_b);
    
  }
  
  ~FooHandler() {
    delete [] h_a;
    delete [] h_b;
    delete h_foo;
    cuda::free(d_a);
    cuda::free(d_b);
    cuda::free(d_foo);
  }
  
  void copy(To::Target target=To::device, int n) {
    cuda::copy(h_a, d_a, n, target);
    cuda::copy(h_b, d_b, n, target);
  }
  
  ...
};

FooHandler *fooHandler = new FooHandler(n);
// for CPU usage
//  fooHandler->h_a;
fooHandler->h_foo;
//  fooHandler->h_foo->a;
// for GPU usage
//  fooHandler->d_a; // e.g. for copying
fooHandler->d_foo; // e.g. as argument for Kernels
//   within kernel class attributes accessible
delete fooHandler;
```

## Integrator

```cpp
class Miluphpc {

public:
  int numParticles;
  int sumParticles;
  int numParticlesLocal;
  int numNodes;

  SPH::KernelHandler kernelHandler;
  SimulationTimeHandler *simulationTimeHandler;
  IntegratedParticleHandler *integratedParticles;
  ParticleHandler *particleHandler;
  SubDomainKeyTreeHandler *subDomainKeyTreeHandler;
  TreeHandler *treeHandler;
  DomainListHandler *domainListHandler;
  DomainListHandler *lowestDomainListHandler;
  MaterialHandler *materialHandler;
  SimulationParameters simulationParameters;

  HelperHandler *buffer;
  
    
  Miluphpc(SimulationParameters simulationParameters);
  ~Miluphpc();

  void prepareSimulation();
  void distributionFromFile(const std::string& filename);

  real removeParticles();
  
  real reset();
  real boundingBox();
  real tree();
  real pseudoParticles();
  real gravity();
  real sph();
  
  real rhs(int step, bool selfGravity=true, bool assignParticlesToProcess=true);

  virtual void integrate(int step = 0) = 0; 
  void afterIntegrationStep();

  real particles2file(int step);
};

class ExplicitEuler : public Miluphpc {

public:
  ExplicitEuler(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {}
  ~ExplicitEuler();
  void integrate(int step);
};

class PredictorCorrectorEuler : public Miluphpc {

public:
  PredictorCorrectorEuler(SimulationParameters simulationParameters) : Miluphpc(simulationParameters) {
    integratedParticles = new IntegratedParticleHandler(numParticles, numNodes);
  }
  ~PredictorCorrectorEuler();
  void integrate(int step);
};

Miluphpc *miluphpc;
// miluphpc = new Miluphpc(parameters, numParticles, numNodes); // not possible since abstract class
switch (parameters.integratorSelection) {
  case IntegratorSelection::explicit_euler: {
    miluphpc = new ExplicitEuler(parameters);
  } break;
  case IntegratorSelection::predictor_corrector_euler: {
    miluphpc = new PredictorCorrectorEuler(parameters);
  } break;
  default: {// exit ...}
}

...
miluphpc->integrate(i_step);
...
```

## Integrated Particles

```cpp
class Particles {

public:
  real *mass;
  real *x;
  ...
};

class IntegratedParticle {

public:
  real *x;
  ...
};

class ParticleHandler {
public:
  real *d_mass = new real[n];
  real *d_x;
  real *_d_x = new int[n];
  ...
  void setPointer(IntegratedParticleHandler *integratedParticleHandler) {
    d_x = integratedParticleHandler->d_x;
    ParticleNS::Kernel::Launch::set(d_mass, d_x, ...);
    ...
    // particleHandler->d_x points to integratedParticleHandler->d_x;
    // particleHandler->d_particles->x points to integratedParticleHandler->d_x;
  }
  void resetPointer() {
    d_x = d_x;
    ParticleNS::Kernel::Launch::set(d_mass, d_x, ...);
    ...
    // particleHandler->d_x points to _d_x;
    // particleHandler->d_particles->x points to _d_x;
  }
};

class IntegratedParticleHandler {
public:
  real *d_x = new real[n];
  ...
};
```

## Different things

### Particle file I/O

```cpp
real Miluphpc::particles2file(int step) {

  boost::mpi::communicator comm;
  sumParticles = numParticlesLocal;
  all_reduce(comm, boost::mpi::inplace_t<integer*>(&sumParticles), 1, std::plus<integer>());
  std::stringstream stepss;
  stepss << std::setw(6) << std::setfill('0') << step;
  HighFive::File h5file(simulationParameters.directory + "ts" + stepss.str() + ".h5", HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate, HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

  std::vector <size_t> dataSpaceDims(2);
  dataSpaceDims[0] = std::size_t(sumParticles);
  dataSpaceDims[1] = DIM;
    
  ...

  HighFive::DataSet h5_pos = h5file.createDataSet<real>("/x", HighFive::DataSpace(dataSpaceDims));
    HighFive::DataSet h5_mass = h5file.createDataSet<real>("/m", HighFive::DataSpace(sumParticles));
  ...
    
  std::vector<std::vector<real>> x;
  std::vector<real> mass;
    
  for (int i=0; i<numParticlesLocal; i++) {
    x.push_back({particleHandler->h_x[i], particleHandler->h_y[i], particleHandler->h_z[i]});
    mass.push_back(particleHandler->h_mass[i]);
    ...
  }

  // receive buffer
  int procN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];

  // send buffer
  int sendProcN[subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses];
  for (int proc=0; proc<subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses; proc++){
    sendProcN[proc] = subDomainKeyTreeHandler->h_subDomainKeyTree->rank == proc ? numParticlesLocal : 0;
  }

  all_reduce(comm, sendProcN, subDomainKeyTreeHandler->h_subDomainKeyTree->numProcesses, procN, boost::mpi::maximum<integer>());

  std::size_t nOffset = 0;
  // count total particles on other processes
  for (int proc = 0; proc < subDomainKeyTreeHandler->h_subDomainKeyTree->rank; proc++){
    nOffset += procN[proc];
  }
    
  h5_pos.select({nOffset, 0}, {std::size_t(numParticlesLocal), std::size_t(DIM)}).write(x);
  h5_mass.select({nOffset}, {std::size_t(numParticlesLocal)}).write(mass);
  ...
}
```

## Reading input file

```cpp
void Miluphpc::distributionFromFile(const std::string& filename) {

  HighFive::File file(filename.c_str(), HighFive::File::ReadOnly);

  std::vector<real> m;
  std::vector<std::vector<real>> x;
  ...

  // read datasets from file
  HighFive::DataSet mass = file.getDataSet("/m");
  HighFive::DataSet pos = file.getDataSet("/x");
  ...

  mass.read(m);
  pos.read(x);
  ...

  int ppp = m.size()/subDomainKeyTreeHandler->h_numProcesses;
  int ppp_remnant = m.size() % subDomainKeyTreeHandler->h_numProcesses;

  int startIndex = subDomainKeyTreeHandler->h_subDomainKeyTree->rank * ppp;
  int endIndex = (subDomainKeyTreeHandler->h_rank + 1) * ppp;
  if (subDomainKeyTreeHandler->h_rank == (subDomainKeyTreeHandler->h_numProcesses - 1)) {
    endIndex += ppp_remnant;
  }

  for (int j = startIndex; j < endIndex; j++) {
    int i = j - subDomainKeyTreeHandler->h_rank * ppp;
    particleHandler->h_particles->mass[i] = m[j];
    particleHandler->h_particles->x[i] = x[j][0];
#if DIM > 1
    particleHandler->h_particles->y[i] = x[j][1];
#if DIM == 3
    particleHandler->h_particles->z[i] = x[j][2];
#endif
#endif
    ...
  }
}
```
