# CMake 


## Links

* [Repository](https://gitlab.kitware.com/cmake/cmake)
* [Awesome-CMake list](https://github.com/onqtam/awesome-cmake)

### Documentation

* [CMake official documentation](https://cmake.org/cmake/help/v3.19/)
	* [CMake commands](https://cmake.org/cmake/help/latest/manual/cmake-commands.7.html)
	* [CMake environmental variables](https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html)
	*  ...
* [The Architecture of Open Source Applications](http://www.aosabook.org/en/cmake.html)

### Tutorials, Guides & Instructions

* [Effective Modern CMake (Dos & Don'ts)](https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1)
* [GitBook: Introduction to Modern CMake](https://cliutils.gitlab.io/modern-cmake/)
* [Official CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
* [CMake User Interaction Guide](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html)
* [CMake Cookbook](https://github.com/dev-cafe/cmake-cookbook)
* [CMake Primer](https://llvm.org/docs/CMakePrimer.html)

### Videos

* [Intro to CMake](https://www.youtube.com/watch?v=HPMvU64RUTY)
* [Using Modern CMake Patterns to Enforce a Good Modular Design](https://www.youtube.com/watch?v=eC9-iRN2b04)
* [Effective CMake](https://www.youtube.com/watch?v=bsXLMQ6WgIk)
* [Embracing Modern CMake](https://www.youtube.com/watch?v=JsjI5xr1jxM)

## Usage

### Command line tools & Interactive Dialogs

#### cmake

The [cmake](https://cmake.org/cmake/help/latest/manual/cmake.1.html) executable is the command-line interface of the cross-platform buildsystem generator *CMake*.

```cmake
# Generate a Project Buildsystem
cmake [<options>] <path-to-source>
cmake [<options>] <path-to-existing-build>
cmake [<options>] -S <path-to-source> -B <path-to-build>
# Build a Project
cmake --build <dir> [<options>] [-- <build-tool-options>]
# Install a Project
cmake --install <dir> [<options>]
# Open a Project
cmake --open <dir>
# Run a Script
cmake [{-D <var>=<value>}...] -P <cmake-script-file>
# Run a Command-Line Tool
cmake -E <command> [<options>]
# Run the Find-Package Tool
cmake --find-package [<options>]
# View Help
cmake --help[-<topic>]
```

#### ctest

The [ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) executable is the CMake test driver program. Build trees using ```enable_testing()``` and ```add_test``` have testing support.

```cmake
ctest [<options>]
ctest --build-and-test <path-to-source> <path-to-build>
      --build-generator <generator> [<options>...]
      [--build-options <opts>...] [--test-command <command> [<args>...]]
ctest {-D <dashboard> | -M <model> -T <action> | -S <script> | -SP <script>}
      [-- <dashboard-options>...]
```

#### cpack

The [cpack](https://cmake.org/cmake/help/latest/manual/cpack.1.html) executable is the CMake packaging program, generating installers and source packages in a variety of formats.

```cmake
cpack [<options>]
```

#### ccmake

The [ccmake](https://cmake.org/cmake/help/latest/manual/ccmake.1.html) executable is the CMake curses interface enabling to configure settings interactively through this GUI.

```cmake
ccmake [<options>] {<path-to-source> | <path-to-existing-build>}
```

#### cmake-gui

The [cmake-gui](https://cmake.org/cmake/help/latest/manual/cmake-gui.1.html) executable is the CMake GUI.

```cmake
cmake-gui [<options>]
cmake-gui [<options>] {<path-to-source> | <path-to-existing-build>}
cmake-gui [<options>] -S <path-to-source> -B <path-to-build>
cmake-gui [<options>] --browse-manual
```

## Basics

### CMake Version

```cmake
#minmum CMake version
cmake_minimum_required(VERSION 3.12)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
```
### VARIABLES

See [CMake variables](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html)

```cmake
# Local variable
set(MY_VARIABLE "value")
set(MY_LIST "one" "two")
# Cache variable
set(MY_CACHE_VARIABLE "VALUE" CACHE STRING "Description")
# Environmental variables
set(ENV{variable_name} value) #access via $ENV{variable_name}
```

### PROPERTIES

See [CMake properties](https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html)

```cmake
set_property(TARGET TargetName PROPERTY CXX_STANDARD 11)
set_target_properties(TargetName PROPERTIES CXX_STANDARD 11)
get_property(ResultVariable TARGET TargetName PROPERTY CXX_STANDARD)
```

### Output folders

```cmake
# set output folders
set(PROJECT_SOURCE_DIR)
set(CMAKE_SOURCE_DIR ...)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}$/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
```

### Sources

```cmake
# set sources
set(SOURCES example.cu)
file(GLOB SOURCES *.cu)
```

### Executables & targets

Add executable/create target:

```cmake
#add_executable(example ${PROJECT_SOURCE_DIR}/example.cu)
add_executable(miluphcuda ${SOURCES})
```

```cmake
# add include directory to target
target_include_directories(miluphcdua PUBLIC include) #PUBLIC/PRIVATE/INTERFACE
# add compile feature to target
target_compile_features(miluphcuda PUBLIC cxx_std_11)
```

```cmake
# chain targets (assume "another" is a target)
add_library(another STATIC another.cpp another.h)
target_link_libraries(another PUBLIC miluphcuda)
```


### PROGRAMMING IN CMAKE

See [the CMake language](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html) and [CMake commands](https://cmake.org/cmake/help/latest/manual/cmake-commands.7.html)

Keywords: 

* NOT
* TARGET
* EXISTS
* DEFINED
* STREQUAL
* AND
* OR
* MATCHES
* ...

#### Control flow

```cmake
if(variable)
    # If variable is `ON`, `YES`, `TRUE`, `Y`, or non zero number
else()
    # If variable is `0`, `OFF`, `NO`, `FALSE`, `N`, `IGNORE`, `NOTFOUND`, `""`, or ends in `-NOTFOUND`
#endif()
```

#### Loops

* ```foreach(var IN ITEMS foo bar baz) ... endforeach()```
* ```foreach(var IN LISTS my_list) ... endforeach()```
* ```foreach(var IN LISTS my_list ITEMS foo bar baz) ... endforeach```
* ```while() ... endwhile()```

Within loops

* ```break()```
* ```continue()```


#### Generator expression

See [CMake generator expressions](https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html)

```cmake
target_include_directories(
        MyTarget
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
```

#### Functions (& macros)

```cmake
function(SIMPLE REQUIRED_ARG)
    message(STATUS "Simple arguments: ${REQUIRED_ARG}, followed by ${ARGV}")
    set(${REQUIRED_ARG} "From SIMPLE" PARENT_SCOPE)
endfunction()
simple(This)
message("Output: ${This}")
```

### COMMUNICATION WITH CODE

#### Configure File

```cmake
configure_file()
...
```

#### Reading files

```cmake
...
```

### RUNNING OTHER PROGRAMS

#### command at configure time

```cmake
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
endif()
```

#### command at build time

```cmake
find_package(PythonInterp REQUIRED)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/Generated.hpp"
        COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/GenerateHeader.py" --argument
        DEPENDS some_target)
add_custom_target(generate_header ALL
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/include/Generated.hpp")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/Generated.hpp DESTINATION include)
```

## Libraries

```cmake
# make a library
add_library(one STATIC two.cpp three.h) # STATIC/SHARED/MODULE
```

## Policies

[CMake policies](https://cmake.org/cmake/help/v3.0/manual/cmake-policies.7.html) is how CMake implements backward compability.

CMake policies know two states

* **old**: makes CMake revert to old behaviour (existed before introduction of the policy)
* **new**: makes CMake use the new behavior that is considered correct and prefered

## Language/Package related

### C

```cmake
# set C compiler
set(CMAKE_C_COMPILER "/usr/bin/gcc-7")
execute_process (
        COMMAND bash -c "git describe --abbrev=4 --dirty --always --tags'"
        OUTPUT_VARIABLE GIT_VERSION
)
```

```cmake
#set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_C_FLAGS "-c -std=c99 -O3 -DVERSION=\"${GIT_VERSION})\" -fPIC")
#set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb")
#set(CMAKE_C_FLAGS_RELEASE "-O0 -ggdb")
```

### C++

...

### CUDA

See [Combining CUDA and Modern CMake](https://on-demand.gputechconf.com/gtc/2017/presentation/S7438-robert-maynard-build-systems-combining-cuda-and-machine-learning.pdf)

#### Enable Cuda support

CUDA is not optional

```cmake
project(MY_PROJECT LANGUAGES CUDA CXX)
```

CUDA is optional

```cmake
enable_language(CUDA)
```

Check whether CUDA is available

```cmake
include(CheckLanguage)
check_language(CUDA)
```

#### CUDA Variables

Exchange *CXX* with *CUDA*

E.g. setting CUDA standard:

```cmake
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
```

#### Adding libraries / executables

As long as *.cu* is used for CUDA files, the procedure is as normal.

With separable compilation

```cmake
set_target_properties(mylib PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
```

#### Architecture

Use ```CMAKE_CUDA_ARCHITECTURES``` variable and the ```CUDA_ARCHITECTURES property``` on targets.

#### Working with targets

Compiler option

```cmake
"$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-fopenmp>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fopenmp>"
```

Use a function that will fix a C++ only target by wrapping the flags if using a CUDA compiler

```cmake
function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
    endif()
endfunction()
```

#### Useful variables

* ```CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES```: Place for built-in Thrust, etc
* ```CMAKE_CUDA_COMPILER```: NVCC with location

```cmake
set(CUDA_DIR "/usr/local/cuda-10.0")
set(CMAKE_CUDA_COMPILER ${CUDA_DIR}/bin/nvcc)
set(CMAKE_CUDA_FLAGS "-ccbin ${CC} -x cu -c -dc -O3  -Xcompiler "-O3 -pthread" -Wno-deprecated-gpu-targets -DVERSION=\"${GIT_VERSION}\"  --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS_DEBUG ...)
set(CMAKE_CUDA_HOST_COMPILER ...)
set(CMAKE_CUDA_EXTENSIONS ...)
set(CMAKE_CUDA_STANDARD ...)
set(CMAKE_CUDA_RUNTIME_LIBRARY ...)
...
```

### OpenMP

#### Enable OpenMP support

```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MyTarget PUBLIC OpenMP::OpenMP_CXX)
endif()
```

### Boost

The Boost library is included in the find packages that CMake provides.

(Common) Settings related to boost

* ```set(Boost_USE_STATIC_LIBS OFF)```
* ```set(Boost_USE_MULTITHREADED ON)```
* ```set(Boost_USE_STATIC_RUNTIME OFF)``

E.g.: using the ```Boost::filesystem``` library

```cmake
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.50 REQUIRED COMPONENTS filesystem)
message(STATUS "Boost version: ${Boost_VERSION}")

# This is needed if your Boost version is newer than your CMake version
# or if you have an old version of CMake (<3.5)
if(NOT TARGET Boost::filesystem)
    add_library(Boost::filesystem IMPORTED INTERFACE)
    set_property(TARGET Boost::filesystem PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
    set_property(TARGET Boost::filesystem PROPERTY
        INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
endif()
```


### MPI

#### Enable MPI support

```cmake
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(MyTarget PUBLIC MPI::MPI_CXX)
```


## Adding features

### Set default build type

```cmake
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
```

### Meta compiler features

```cmake
target_compile_features(myTarget PUBLIC cxx_std_11)
set_target_properties(myTarget PROPERTIES CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_target_properties(myTarget PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
```

### Position independent code (-fPIC)

```cmake
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# or target dependent
set_target_properties(lib1 PROPERTIES POSITION_INDEPENDENT_CODE ON)
```

### Little libraries

```cmake
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(MyTarget PUBLIC ${MATH_LIBRARY})
endif()
```

### Modules

See [CMake modules](https://cmake.org/cmake/help/latest/manual/cmake-modules.7.html)

#### CMakeDependentOption

```cmake
include(CMakeDependentOption)
cmake_dependent_option(BUILD_TESTS "Build your tests" ON "VAL1;VAL2" OFF)
```

which is equivalent to

```cmake
if(VAL1 AND VAL2)
    set(BUILD_TESTS_DEFAULT ON)
else()
    set(BUILD_TESTS_DEFAULT OFF)
endif()

option(BUILD_TESTS "Build your tests" ${BUILD_TESTS_DEFAULT})

if(NOT BUILD_TESTS_DEFAULT)
    mark_as_advanced(BUILD_TESTS)
endif()
```

#### CMakePrintHelpers

```cmake
cmake_print_properties
cmake_print_variables
```

#### CheckCXXCompilerFlag

Check whether flag is supported

```cmake
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-someflag OUTPUT_VARIABLE)
```

#### WriteCompilerDetectionHeader

Look for a list of features that some compilers support and write out a C++ header file that lets you know whether that feature is available

```cmake
write_compiler_detection_header(
  FILE myoutput.h
  PREFIX My
  COMPILERS GNU Clang MSVC Intel
  FEATURES cxx_variadic_templates
)
```

#### try\_compile / try\_run

```cmake
try_compile(
  RESULT_VAR
    bindir
  SOURCES
    source.cpp
)
```

## Debugging

### Printing variables

```cmake
message(STATUS "MY_VARIABLE=${MY_VARIABLE}")
# or using module 
include(CMakePrintHelpers)
cmake_print_variables(MY_VARIABLE)
cmake_print_properties(
    TARGETS my_target
    PROPERTIES POSITION_INDEPENDENT_CODE
)
```

### Tracing a run

```cmake
cmake -S . -B build --trace-source=CMakeLists.txt #--trace-expand 
```

## Including projects

### Fetch

E.g.: download Catch2

```cmake
FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.0
)
# CMake 3.14+
FetchContent_MakeAvailable(catch)
```

## Testing

### General

Enable testing and set a ```BUILD_TESTING``` option

```cmake
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()
```

Add test folder

```cmake
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
```

Register targets

```cmake
add_test(NAME TestName COMMAND TargetName)
add_test(NAME TestName COMMAND $<TARGET_FILE:${TESTNAME}>)
```

### Building as part of the test

```cmake
add_test(
  NAME
    ExampleCMakeBuild
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
             --build-and-test "${My_SOURCE_DIR}/examples/simple"
                              "${CMAKE_CURRENT_BINARY_DIR}/simple"
             --build-generator "${CMAKE_GENERATOR}"
             --test-command "${CMAKE_CTEST_COMMAND}"
)
```

### Testing frameworks

#### GoogleTest

See [Modern CMake: GoogleTest](https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html) for reference.

Checkout GoogleTest as submodule

```cmake
git submodule add --branch=release-1.8.0 ../../google/googletest.git extern/googletest
```

```cmake
option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()
```

#### Catch2

```cmake
# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/catch)
add_library(Catch2::Catch IMPORTED INTERFACE)
set_property(Catch2::Catch PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CATCH_INCLUDE_DIR}")
```

#### DocTest 

*DocTest* is a replacement for *Catch2* that is supposed to compile much faster and be cleaner. Just replace *Catch2* with *DocTest*.

## Exporting and Installing

Allow others to use your library, via

* *Bad way:* Find module
* *Add subproject:* ```add_library(MyLib::MyLib ALIAS MyLib)```
* *Exporting:* Using *Config.cmake scripts

### Installing

See [GitBook: Installing](https://cliutils.gitlab.io/modern-cmake/chapters/install/installing.html)
Basic target install command (executed by e.g. ```make install```)

```cmake
install(TARGETS MyLib
        EXPORT MyLibTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
```

### Exporting

See [GitBook: Exporting](https://cliutils.gitlab.io/modern-cmake/chapters/install/exporting.html)

### Packaging

See [GitBook: Packaging](https://cliutils.gitlab.io/modern-cmake/chapters/install/packaging.html)







