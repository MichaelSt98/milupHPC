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

	
## Kernel skeleton

```cpp

//*.cuh
namespace Kernel {
	__global__ void skeleton(Class *class, integer n);

	namespace Launch {
		void skeleton(Class *class, integer n);
	}
}

//*.cu
__global__ void Kernel::skeleton(Class *class, integer n) {

	integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
    integer stride = blockDim.x * gridDim.x;
    
    while ((bodyIndex + offset) < n) {
		
		class->x[bodyIndex + offset] = 0;
		    
    	offset += stride;
    }
}

void Kernel::Launch::skeleton(Class *class, integer n) {
	ExecutionPolicy executionPolicy();
	cuda::launch(false, executionPolicy, ::Kernel::skeleton, class, n);
}

//*.cpp

Kernel::Launch::skeleton(class, n);




```

## Integrator

```cpp
#include "IntegratedParticles.cuh"

struct Integrator
{
    enum Type
    {
        euler, predictor_corrector
    };
    Type t_;
    Integrator(Type t) : t_(t) {}
    operator Type () const {return t_;}
private:
    template<typename T>
    operator T () const;
};

class Integrator {

public:

	BaseIntegrator *baseIntegrator;
	
	Integrator(Integrator::Type integratorType) {
	
		switch(integratorType) {
			case euler:
				baseIntegrator = new Euler();
				break;
			// ...
		}
	
	}
	
	void integrate() {
		baseIntegrator->integrate();
	}

}

class BaseIntegrator {
	
	//...
	
	virtual void integrate();
	void rhs() {
	
	}
}

class Euler < BaseIntegrator {
	
public:
	IntegratedParticles *integratedParticles_1;
	
	void integrate() {
		rhs();
		rhs();
		rhs();
	}
	
	Euler() {
		
	}
}
```


## Material

* include all possible parameters
	 * no *memory optimization* 

```cpp

materials = new Material[10];

for () {
	materials[i].config()
}

materials[particles[index].materialId].a;

class Material {
public:

	integer ID = 0
    char *name = "Basalt porous (Tillotson)"
    real sml = 0.78
    integer interactions = 30
    struct artificial_viscosity { 
    	alpha = 1.0; 
    	beta = 2.0; 
    };
    real density_floor = 100.
    struct eos {
        integer type = 5
        real shear_modulus = 22.7e9
        real bulk_modulus = 26.7e9
        // ...
        //# include Tillotson EoS params
        //@include "material_data/basalt.till.cfg"
    };
    
    struct porosity {
    	porjutzi_p_elastic = 1.0e6
        porjutzi_p_transition = 6.80e7
        porjutzi_p_compacted = 2.13e8
        porjutzi_alpha_0 = 2.0
        porjutzi_alpha_e = 4.64
        porjutzi_alpha_t = 1.90
        porjutzi_n1 = 12.0
        porjutzi_n2 = 3.0
        //...
    };
    
    struct plasticity {
    	yield_stress = 3.5e9
	    cohesion = 1.0e6
   	    friction_angle = 0.9827937232
   	    friction_angle_damaged = 0.5404195003
   	    // ...
    };
    
    Material(configFile, ID);
}
```
