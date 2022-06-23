# Binac 

## Links

* [bwhpc](https://wiki.bwhpc.de/e/Main_Page)

## Accessing

* Login: 
	* ssh 
		* `ssh <UserID>@login01.binac.uni-tuebingen.de`
		* `ssh <UserID>@login01.binac.uni-tuebingen.de`
		* `ssh <UserID>@login01.binac.uni-tuebingen.de`
	* generate and enter Code using App (FreeOTP)
	* enter password
	* navigate to directory (e.g.: `cd /beegfs/work/tu_zxmjo49`)


## Compiling

### Modules

* `$ man module` for more information
	* loaded modules: `module list`
	* available modules: `module avail` 

* modules to load:
	* `module load devel/cuda/10.1`
	* `module load mpi/openmpi/3.1-gnu-8.3` 

### Linking

```
CXXFLAGS    := -std=c++11 -I/opt/bwhpc/common/lib/hdf5/1.10.7-openmpi-3.1-gnu-9.2/include -I/opt/bwhp
c/common/lib/boost/1.69.0/include
```

```
LFLAGS      := -std=c++11 -L/opt/bwhpc/common/lib/hdf5/1.10.7-openmpi-3.1-gnu-9.2/lib -L/opt/bwhpc/co
mmon/lib/boost/1.69.0/lib -lboost_filesystem -lboost_system -lhdf5  #-L/usr/lib/x86_64-linux-gnu/hdf5
/openmpi -lhdf5
```

```
module load compiler/gnu/9.2
module load lib/boost/1.76.0
module load devel/cuda/10.1
module load mpi/openmpi/4.1-gnu-9.2
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
```

### Batch job

Used Queing system: [OpenPBS](https://www.openpbs.org/)

* `#PBS -l nodes=1:ppn=4:gpus=4:default` for four GPUs on one node
	* `export CUDA_VISIBLE_DEVICES=0,1,2,3`

```
#!/bin/bash
#PBS -N N1e6SSG
#PBS -l nodes=1:ppn=2
#PBS -l walltime=00:02:00
#PBS -l pmem=1gb
#PBS -q tiny
#PBS -m aeb -M johannes-stefan.martin@student.uni-tuebingen.de
source ~/.bashrc

# Loading modules
module load mpi/openmpi/3.1-gnu-9.2
module load lib/hdf5/1.10.7-openmpi-3.1-gnu-9.2
module load lib/boost/1.69.0

# Going to working directory
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

# Starting program
mpirun --bind-to core --map-by core -report-bindings bin/runner
```


```
#Compiler/Linker
CXX         := mpic++ #g++

#Target binary
TARGET      := runner

#Directories
SRCDIR      := ./src
INCDIR      := ./include
BUILDDIR    := ./build
TARGETDIR   := ./bin
RESDIR      := ./resources
IDEASDIR    := ./ideas
TESTDIR     := ./test
DOCDIR      := ./doc
DOCUMENTSDIR:= ./documents
IMAGESDIR := ./images
OUTDIR := ./output

SRCEXT      := cpp
DEPEXT      := d
OBJEXT      := o

#Flags, Libraries and Includes
CXXFLAGS    := -std=c++11 -I/opt/bwhpc/common/lib/hdf5/1.10.7-openmpi-3.1-gnu-9.2/include -I/opt/bwhpc/common/lib/boost/1.76.0/include #-I/usr/include/hdf5/openmpi #-Wno-conversion -g3 #-std=c++17 -O3 -Wall -pedantic -Wno-vla-extension -I/usr/local/include/ -I/usr/local/include/eigen3/ -I./include -I./src
LFLAGS      := -std=c++11 -L/opt/bwhpc/common/lib/hdf5/1.10.7-openmpi-3.1-gnu-9.2/lib -L/opt/bwhpc/common/lib/boost/1.76.0/lib -lboost_filesystem -lboost_system -lhdf5  #-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -lhdf5 #-O3 -Wall -Wno-deprecated -Werror -pedantic -L/usr/local/lib/
LIB         := -Xpreprocessor -fopenmp #-lomp
INC         := -I$(INCDIR) #-I/usr/local/include
INCDEP      := -I$(INCDIR)

#Source and Object files
SOURCES     := $(shell find $(SRCDIR) -type f -name "*.$(SRCEXT)")
OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

#Documentation (Doxygen)
DOXY        := /usr/local/Cellar/doxygen/1.8.20/bin/doxygen
DOXYFILE    := $(DOCDIR)/Doxyfile

#default make (all)
all: resources tester ideas $(TARGET)

# build all with debug flags
debug: CXXFLAGS += -g
# show linker invocation when building debug target
debug: LDFLAGS += -v
debug: all

#make regarding source files
sources: resources $(TARGET)

#remake
remake: cleaner all

#copy Resources from Resources Directory to Target Directory
resources: directories
	@cp -r $(RESDIR)/ $(TARGETDIR)/

#make directories
directories:
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)

#clean objects
clean:
	@$(RM) -rf $(BUILDDIR)

#clean objects and binaries
cleaner: clean
	@$(RM) -rf $(TARGETDIR)

#creating a new video of the particle N-body simulation
movie: $(TARGET) | $(IMAGESDIR)
	@echo "Creating video ..."
	@rm -f $(IMAGESDIR)/*.ppm
	@mpirun -np 2 bin/runner
	@./createMP4
	@echo "... done."

h5files: $(TAGRET) | $(OUTDIR)
	@echo "Creating h5-files in $(OUTDIR) ..."
	@rm -f $(OUTDIR)/*.h5
	@mpirun -np 2 bin/runner
	@echo "... done."

$(IMAGESDIR):
	@mkdir -p $@

#Pull in dependency info for *existing* .o files
-include $(OBJECTS:.$(OBJEXT)=.$(DEPEXT)) #$(INCDIR)/matplotlibcpp.h

#link
$(TARGET): $(OBJECTS)
	@echo "Linking ..."
	@$(CXX) $(LFLAGS) $(INC) -o $(TARGETDIR)/$(TARGET) $^ $(LIB)

#compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@echo "  compiling: " $(SRCDIR)/$*
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INC) -c -o $@ $< $(LIB)
	@$(CXX) $(CXXFLAGS) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp


#compile test files
tester: directories
ifneq ("$(wildcard $(TESTDIR)/*.$(SRCEXT) )","")
	@echo "  compiling: " test/*
	@$(CXX) $(CXXFLAGS) test/*.cpp $(INC) $(LIB) -o bin/tester
else
	@echo "No $(SRCEXT)-files within $(TESTDIR)!"
endif


#compile idea files
ideas: directories
ifneq ("$(wildcard $(IDEASDIR)/*.$(SRCEXT) )","")
	@echo "  compiling: " ideas/*
	@$(CXX) $(CXXFLAGS) ideas/*.cpp $(INC) $(LIB) -o bin/ideas
else
	@echo "No $(SRCEXT)-files within $(IDEASDIR)!"
endif

doxyfile.inc: #Makefile
	@echo INPUT            = README.md . $(SRCDIR)/ $(INCDIR)/ $(DOCUMENTSDIR)/ > $(DOCDIR)/doxyfile.inc
	@echo FILE_PATTERNS     = "*.md" "*.h" "*.$(SRCEXT)" >> $(DOCDIR)/doxyfile.inc
	@echo OUTPUT_DIRECTORY = $(DOCDIR)/ >> $(DOCDIR)/doxyfile.inc

doc: doxyfile.inc
	$(DOXY) $(DOXYFILE) &> $(DOCDIR)/doxygen.log
	@$(MAKE) -C $(DOCDIR)/latex/ &> $(DOCDIR)/latex/latex.log
	@mkdir -p "./docs"
	cp -r "./doc/html/" "./docs/"

#Non-File Targets
.PHONY: all remake debug clean cleaner resources sources directories ideas tester doc movie h5files
```

```
nvcc -arch=sm_52 -I/opt/bwhpc/common/lib/boost/1.69.0/include -I./include -I/home/tu/tu_tu/tu_zxmjo49/local/include -I/opt/bwhpc/common/devel/cuda/10.1/include -lmpi -lhdf5 -lboost_filesystem -lboost_system -o ./bin/runner build/materials/material_handler.o build/miluphpc.o build/utils/config_parser.o build/utils/h5profiler.o build/utils/timer.o build/utils/logger.o build/particle_handler.o build/main.o build/subdomain_key_tree/subdomain_handler.o build/subdomain_key_tree/tree_handler.o build/cuda_utils/cuda_runtime.o build/helper_handler.o build/integrator/euler.o build/integrator/explicit_euler.o build/integrator/predictor_corrector.o build/materials/material.o build/particles.o build/helper.o build/gravity/gravity.o build/device_rhs.o build/subdomain_key_tree/tree.o build/subdomain_key_tree/subdomain.o build/cuda_utils/cuda_utilities.o build/cuda_utils/cuda_launcher.o build/cuda_utils/linalg.o build/sph/sph.o build/sph/kernel_handler.o build/sph/density.o build/sph/kernel.o build/sph/pressure.o  -lm -L/opt/bwhpc/common/devel/cuda/10.1/lib64 -L/home/tu/tu_tu/tu_zxmjo49/local/lib -lcudart -lpthread -lconfig -L/usr/local/cuda-11.4/lib64 -L/opt/openmpi-4.1.0/lib -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi
```




