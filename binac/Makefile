#Compiler/Linker
CXX            := mpic++
NVCC           := nvcc

#Target binary
TARGET         := runner

#Directories
SRCDIR         := ./src
INCDIR         := ./include
BUILDDIR       := ./build
TARGETDIR      := ./bin
RESDIR         := ./resources
IDEASDIR       := ./ideas
TESTDIR        := ./test
DOCDIR         := ./doc
DOCUMENTSDIR   := ./documents
CUDADIR        := /opt/bwhpc/common/devel/cuda/10.1

SRCEXT         := cpp
CUDASRCEXT     := cu
DEPEXT         := d
OBJEXT         := o

#Flags, Libraries and Includes
CXXFLAGS       += -std=c++11 -w -I/opt/bwhpc/common/mpi/openmpi/4.1.1-gnu-8.3/include -I/opt/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-8.3/include -I${HOME}/local/include#-I/usr/include/hdf5/openmpi#-O3
NVFLAGS        := --std=c++11 -x cu -c -dc -w -Xcompiler "-pthread" -Wno-deprecated-gpu-targets -O3 -L/opt/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-8.3/lib -L${HOME}/lib -L/opt/bwhpc/common/mpi/openmpi/4.1.1-gnu-8.3/lib -I/opt/bwhpc/common/mpi/openmpi/4.1.1-gnu-8.3/include  -lboost_filesystem -lboost_system -lhdf5#-I/opt/openmpi-4.1.0/include -I/usr/include/hdf5/openmpi
LFLAGS         += -lm -L$(CUDADIR)/lib64 -L${HOME}/local/lib -lcudart -lpthread -lconfig -L/opt/bwhpc/common/devel/cuda/10.1/lib -L/opt/bwhpc/common/mpi/openmpi/4.1.1-gnu-8.3/lib -L${HOME}/local/lib -L/opt/bwhpc/common/lib/hdf5/1.12.0-openmpi-4.1-gnu-8.3/lib -lboost_atomic -lboost_serialization -lboost_mpi -lmpi -lhdf5 -lboost_filesystem -lboost_system
GPU_ARCH       := -arch=sm_37#sm_52
CUDALFLAGS     := -dlink
CUDALINKOBJ    := cuLink.o #needed?
LIB            := #-lboost_mpi -lboost_serialization
INC            := -I/opt/bwhpc/common/lib/boost/1.69.0/include -I$(INCDIR) -I${HOME}/local/include -I$(CUDADIR)/include #-I/opt/openmpi-4.1.0/include -I/usr/local/Headeronly/ #-L/opt/openmpi-4.1.0/lib -lmpi #-I/usr/local/include
INCDEP         := -I$(INCDIR)

#Source and Object files
SOURCES        := $(shell find $(SRCDIR) -type f -name "*.$(SRCEXT)")
CUDA_SOURCES   := $(shell find $(SRCDIR) -type f -name "*.$(CUDASRCEXT)")
OBJECTS        := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))
CUDA_OBJECTS   := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CUDA_SOURCES:.$(CUDASRCEXT)=.$(OBJEXT)))

#Documentation (Doxygen)
DOXY           := /usr/local/Cellar/doxygen/1.8.20/bin/doxygen
DOXYFILE       := $(DOCDIR)/Doxyfile

#default make (all)
all:  tester ideas $(TARGET)

single-precision: CXXFLAGS += -DSINGLE_PRECISION
single-precision: NVFLAGS += -DSINGLE_PRECISION
single-precision: all

debug: CXXFLAGS += -g
debug: NVFLAGS  += -g -G
debug: tester ideas $(TARGET)

#make regarding source files
sources: resources $(TARGET)

#remake
remake: cleaner all

#copy Resources from Resources Directory to Target Directory
resources: directories
	@cp -r $(RESDIR)/ $(TARGETDIR)/

#make directories
directories:
	@mkdir -p $(RESDIR)
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)

#clean objects
clean:
	@$(RM) -rf $(BUILDDIR)

#clean objects and binaries
cleaner: clean
	@$(RM) -rf $(TARGETDIR)

#Pull in dependency info for *existing* .o files
-include $(OBJECTS:.$(OBJEXT)=.$(DEPEXT)) #$(INCDIR)/matplotlibcpp.h

#link
$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	@echo "Linking ..."
	$(NVCC) $(GPU_ARCH) $(INC) -o $(TARGETDIR)/$(TARGET) $^ $(LIB) $(LFLAGS) #$(GPU_ARCH)

#compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@echo "  compiling: " $(SRCDIR)/$*
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INC) -c -o $@ $< $(LIB)
	@$(CXX) $(CXXFLAGS) $(INC) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(CUDASRCEXT)
	@echo "  compiling: " $(SRCDIR)/$*
	@mkdir -p $(dir $@)
	@$(NVCC) $(GPU_ARCH) $(INC) $(NVFLAGS) -I$(CUDADIR) -c -o $@ $<
	@$(NVCC) $(GPU_ARCH) $(INC) $(NVFLAGS) -I$(CUDADIR) -MM $(SRCDIR)/$*.$(CUDASRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)

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

#@echo FILE_PATTERNS     = "*.md" "*.h" "*.$(SRCEXT)" >> $(DOCDIR)/doxyfile.inc
doxyfile.inc: #Makefile
	@echo INPUT            = README.md . $(SRCDIR)/ $(INCDIR)/ $(DOCUMENTSDIR)/ > $(DOCDIR)/doxyfile.inc
	@echo OUTPUT_DIRECTORY = $(DOCDIR)/ >> $(DOCDIR)/doxyfile.inc

#@$(MAKE) -C $(DOCDIR)/latex/ &> $(DOCDIR)/latex/latex.log
doc: doxyfile.inc
	$(DOXY) $(DOXYFILE) &> $(DOCDIR)/doxygen.log
	@mkdir -p "./docs"
	cp -r "./doc/html/" "./docs/"

#Non-File Targets
.PHONY: all remake clean cleaner resources sources directories ideas tester doc

