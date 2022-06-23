# Set shell
SHELL := /bin/bash

# Set compiler
CXX := g++

TRGTDIR := bin
TARGET := h5renderer

SRCEXT := cpp
OBJEXT := o
HEADEXT := h
DEPEXT := d

# global include
INCLDIR := include

SRCDIR := src
OBJDIR := obj
DEPDIR := dep

# header only libraries
HOLIB := holib

SOURCES := $(wildcard $(SRCDIR)/*.$(SRCEXT))
DEPENDS := $(SOURCES:$(SRCDIR)/%.$(SRCEXT)=$(DEPDIR)/%.$(DEPEXT))
OBJECTS := $(SOURCES:$(SRCDIR)/%.$(SRCEXT)=$(OBJDIR)/%.$(OBJEXT))

# main source file stem name
MAIN := main

# setting general flags
CXXFLAGS := -std=c++11 -O3 -I$(INCLDIR) -I$(HOLIB) -I/usr/include/hdf5/serial -Xpreprocessor -fopenmp
# setting linker flags
LDFLAGS := -std=c++11 -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5 -lboost_system -lboost_filesystem

# get kernel name to be able to run sed correctly on Darwin (MacOS) or Linux kernels
KERNEL := $(shell uname -s)
ifeq ($(KERNEL), Darwin)
	SED := sed -i "~"
	LDFLAGS += -lomp
else
	SED := sed -i
	LDFLAGS += -lgomp
endif

all: $(OBJECTS) $(TRGTDIR)/$(TARGET)

# build all with debug flags
debug: CXXFLAGS += -g -Wall
# show linker invocation when building debug target
debug: LDFLAGS += -v
debug: all

clean: | $(TRGTDIR)
	@rm -r $(TRGTDIR)/$(TRGT) $(OBJECTS)

cleaner: clean | $(TRGTDIR)
	@rm -r $(DEPENDS)

$(TRGTDIR):
	@mkdir -p $@

$(DEPDIR):
	@mkdir -p $@

$(OBJDIR):
	@mkdir -p $@

$(HOLIB):
	@mkdir -p $@

# generating dependency files for all sources
# sed changes '%.o: ...' to '%.o %.d: ...' in dependency file
$(DEPDIR)/%.$(DEPEXT): $(SRCDIR)/%.$(SRCEXT) | $(DEPDIR)
	@echo "Generating dependency file '$@' ..."
	@$(CXX) -MM $(CXXFLAGS) $< -MF $@
	@$(SED) 's,$(*F).$(OBJEXT),$*.$(OBJEXT) $@,' $@
	@rm -f $@~
	@echo "... done."

# include targets from generated dependency files
include $(DEPENDS)

prepare-run:
	@mkdir -p output

# build module as shared library and install it on your system
install: prepare-run | $(HOLIB)
	@echo "Downloading cxxopts.hpp into holib/ ..."
	curl https://raw.githubusercontent.com/jarro2783/cxxopts/master/include/cxxopts.hpp -o $(HOLIB)/cxxopts.hpp
	@echo "... done. Downloading HighFive as zip-file ..."
	@mkdir tmp
	curl https://github.com/BlueBrain/HighFive/archive/refs/heads/master.zip -L -o tmp/HighFive.zip
	@echo "... extracting and copying include/highfive to holib/ ..."
	@unzip -a tmp/HighFive.zip -d tmp/
	@cp -r tmp/HighFive-master/include/highfive $(HOLIB)/.
	@echo "... done. Deleting tmp/ ..."
	@ rm -rf tmp
	@echo "... done. Ready to build via 'make all|debug|binac'."

binac-load-modules:
	@echo "Loading modules ..."
	source binacLoadModules.sh
	@echo "... done."

binac: CXXFLAGS += -I/opt/bwhpc/common/lib/boost/1.76.0/include -I/opt/bwhpc/common/lib/hdf5/1.10.7-gnu-9.2/include
binac: LDFLAGS := -std=c++11 -L/opt/bwhpc/common/lib/boost/1.76.0/lib -L/opt/bwhpc/common/lib/hdf5/1.10.7-gnu-9.2/lib -lhdf5 -lboost_filesystem -lhdf5 -fopenmp
binac: binac-load-modules all

# build main target
# check if target directory 'bin' already exist via prerequisite
$(TRGTDIR)/$(TARGET): $(OBJDIR)/$(MAIN).$(OBJEXT) $(OBJECTS) | $(TRGTDIR)
	@echo "Linking binary '$@'..."
	@$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TRGTDIR)/$(TARGET) $(LDFLAGS)
	@echo "... done."

# pattern rule to build object from source file
$(OBJDIR)/%.$(OBJEXT): $(DEPDIR)/%.$(DEPEXT) | $(OBJDIR)
	@echo "Compiling '$@' ..."
	@$(CXX) -c $(CXXFLAGS) $(@:$(OBJDIR)/%.$(OBJEXT)=$(SRCDIR)/%.$(SRCEXT)) -o $@
	@echo "... done."

.PHONY: all debug clean cleaner install prepare-run