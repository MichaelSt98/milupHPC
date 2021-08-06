## Features of miluphcuda and their importance/priority

This overview should be the help to prioritize the order of implementations in the new code.
The `#defines` are taken from 

* miluphcuda devel version from timestamp *2021-03-26*


---- 

### Sorted by priority

#### Prio0

* SOLID
* HYDRO
* INTEGRATE\_ENERGY
* INTEGRATE\_DENSITY
* FRAGMENTATION
* DAMAGE\_ACTS\_ON\_S
* SPH\_EQU\_VERSION
* ARTIFICIAL\_STRESS
* ARTIFICIAL\_VISCOSITY
* SHEPARD\_CORRECTION & TENSORIAL\_CORRECTION
* **Plasticity models**
	* VON\_MISES\_PLASTICITY 
* **Porosity models**
	* PALPHA\_POROSITY 
* VARIABLE\_SML
* FIXED\_NOI
* INTEGRATE\_SML
* READINITIAL\_SML\_FROMPARTICLE_FILE
* AVERAGE_KERNELS


#### Prio1

* NAVIER\_STOKES
* SHAKURA\_SUNYAEV\_ALPHA
* CONSTANT\_KINEMATIC\_VISCOSITY
* KLEY\_VISCOSITY
* BALSARA\_SWITCH
* **Plasticity models**
	* DRUCKER\_PRAGER\_PLASTICITY
	* MOHR\_COULOMB\_PLASTICITY
	* COLLINS\_PLASTICITY
	* COLLINS\_PLASTICITY\_INCLUDE\_MELT\_ENERGY
	* COLLINS\_PLASTICITY\_SIMPLE
* **Porosity models**
	* STRESS\_PALPHA\_POROSITY 
	* EPSALPHA\_POROSITY 
* SML\_CORRECTION
* XSPH


#### Prio2

* GRAVITATING\_POINT\_MASSES
* PARTICLE\_ACCRETION
* UPDATE\_SINK\_VALUES
* INVISCID\_SPH


#### Prio3

* BOUNDARY\_PARTICLE\_ID
* GHOST\_BOUNDARIES


#### Prio4

* **Plasticity models**
	* VISCOUS\_REGOLITH
	* PURE\_REGOLITH
	* JC\_PLASTICITY
* **Porosity models**
	* SIRONO\_POROSITY 


#### Not needed

* REAL\_HYDRO

----

### Randomly sorted with explanation

**SOLID** 

* <span style="color:red">prio0</span>

simulate solid body behaviour, i.e. stress tensor is given by $\sigma^{ab} = -p \delta^{ab} + S^{ab}$
particle specific values are: $\vec{x}$, $\vec{v}$, dim $x$ dim matrix $\sigma$ and $S$ where one can save some memory since
S is symmetric and traceless, so in 3 dim, one needs only 5 doubles instead of 9.

**HYDRO** 

* <span style="color:red">prio0</span>

solve the standard inviscid Euler equations, no shear forces


**REAL_HYDRO**

* <span style="color:orange">not needed</span>

allow only positive pressure

**GRAVITATING\_POINT\_MASSES** 

* <span style="color:red">prio2</span>

adds additional point masses to the simulation which interact gravitationally with themselves and with the sph particles imho

> we do not have more than 2 or 3 additional point masses, so no need for parallelization here, probably fastest to
integrate them on all nodes individually

**PARTICLE_ACCRETION** 

* <span style="color:red">prio2</span> 

requires gravitating point masses of boundary conditions. not needed from the very beginning. however, keep in mind that we need to be able to remove and add particles to the simulation.

**UPDATE\_SINK\_VALUES**

* <span style="color:red">prio2</span>


depends on **PARTICLE ACCRETION** 

**INTEGRATE_ENERGY**

* <span style="color:red">prio0</span> 

integrate energy equation, we need this from the very beginning

**INTEGRATE_DENSITY**
 
* <span style="color:red">prio0</span> 

integrate continuity equation, we need this from the very beginning

**NAVIER_STOKES**

* <span style="color:red">prio1</span> 

solve the Navier Stokes equation, viscous flows

**SHAKURA\_SUNYAEV\_ALPHA**

* <span style="color:red">prio1</span> 

depends on NAVIER_STOKES

**CONSTANT\_KINEMATIC\_VISCOSITY**

* <span style="color:red">prio1</span> 

depends on NAVIER_STOKES

**KLEY_VISCOSITY**

* <span style="color:red">prio1</span> 

depends on NAVIER_STOKES

**FRAGMENTATION**

* <span style="color:red">prio0</span> 

damage model following Benz & Asphaug 1994, essential to model rocks, depends on SOLID

**DAMAGE\_ACTS\_ON\_S**

* <span style="color:red">prio0</span> 

depends on FRAGMENTATION

**SPH\_EQU\_VERSION**

* <span style="color:red">prio0</span> 

we've implemented two different versions of the SPH equations, ask cms for reference or lecture notes

**ARTIFICIAL\_STRESS**

* <span style="color:red">prio0</span> 

artificial stress to overcome the tensile instability, required for fully elastic solids (rubber ring collision simulation)

**ARTIFICIAL\_VISCOSITY**

* <span style="color:red">prio0</span> 

the one and only artificial viscosity by Monaghan (see review from 1992), we need this from the very beginning

**BALSARA\_SWITCH** 

* <span style="color:red">prio1</span> 

reduce artificial viscosity where not needed

**INVISCID\_SPH**

* <span style="color:red">prio2</span> 

modern version of the balsara switch with shock capture and time dependent artificial viscosity coefficients

**SHEPARD\_CORRECTION &
TENSORIAL\_CORRECTION**

* <span style="color:red">prio0</span> 

improve standard SPH to provide zeroth order and linear consistency. see standard sph textbooks or latest code paper

**VON\_MISES\_PLASTICITY \
DRUCKER\_PRAGER\_PLASTICITY \
MOHR\_COULOMB\_PLASTICITY \
COLLINS\_PLASTICITY \
COLLINS\_PLASTICITY\_INCLUDE\_MELT\_ENERGY \
COLLINS\_PLASTICITY\_SIMPLE**

* <span style="color:red">prio0-1</span> 

these are all models for plasticity. Eventually we need all of them, for starters VON_MISES_PLASTICITY will do. 

**VISCOUS\_REGOLITH \
PURE\_REGOLITH \
JC\_PLASTICITY**

* <span style="color:red">prio4</span> 

these are all models for plasticity which we currently do not use

**PALPHA\_POROSITY \         
STRESS\_PALPHA\_POROSITY \
EPSALPHA\_POROSITY**

* <span style="color:red">prio0-1</span> 

these are all models for porosity, p-alpha is *prio0*, epsilon alpha is *prio1* and

**SIRONO\_POROSITY**

* <span style="color:red">prio4</span> 

Sirono is currently not needed.


**VARIABLE\_SML \
FIXED\_NOI \
INTEGRATE\_SML\
READ_INITIAL\_SML\_FROM_PARTICLE\_FILE**

* <span style="color:red">prio0</span>


**SML\_CORRECTION**

* <span style="color:red">prio1</span> 

correction factors for variable smoothing length. documented in Evita's master thesis and references therein, or see PHANTOM paper

**AVERAGE\_KERNELS**

* <span style="color:red">prio0</span> 

for varying smoothing length, the values of the kernels are averaged and not the smoothing lengths 


**XSPH**

* <span style="color:red">prio1</span> 

additional smoothing of the velocity field. required for KLEY\_VISCOSITY, nice to have but not prio0

**BOUNDARY\_PARTICLE\_ID \
GHOST\_BOUNDARIES**

* <span style="color:red">prio3</span> 

boundary particles. a chapter on its own. leave it be for starters
