cmake_minimum_required(VERSION 3.13)
project(polymerpim_wrapper CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find JlCxx (CxxWrap C++ library) ----
find_package(JlCxx REQUIRED)
get_target_property(JLCXX_LOCATION JlCxx::cxxwrap_julia LOCATION)
message(STATUS "Found JlCxx at: ${JLCXX_LOCATION}")

# ---- PolymerPIM installation path ----
set(POLYMERPIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../install"
    CACHE PATH "PolymerPIM installation prefix")

if(NOT EXISTS "${POLYMERPIM_ROOT}/include/polymerpim/polymerpim.h")
    message(FATAL_ERROR
        "PolymerPIM not found at ${POLYMERPIM_ROOT}.\n"
        "Build and install it first:\n"
        "  make install BACKEND=hw JIT=1\n"
        "Or set -DPOLYMERPIM_ROOT=/path/to/install")
endif()

# ---- UPMEM SDK (for transitive DPU library dependency) ----
# libpolymerpim already carries the DPU runtime dependency.

# ---- Build the wrapper shared library ----
add_library(polymerpim_wrapper SHARED wrapper.cpp)

target_include_directories(polymerpim_wrapper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../host
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common
)

target_link_directories(polymerpim_wrapper PRIVATE
    ${POLYMERPIM_ROOT}/lib
    ${UPMEM_HOME}/lib
)

target_link_libraries(polymerpim_wrapper
    JlCxx::cxxwrap_julia
    polymerpim
)

# No install() rule: deps/build.jl copies the artifact into PolymerPIM/ here,
# which keeps the RPATH cmake would otherwise strip on install.
