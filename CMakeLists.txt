cmake_minimum_required(VERSION 3.2.0)
project(stream_cuda CXX)

set(VERSION_MAJOR "0")
set(VERSION_MINOR "2")
set(VERSION_PATCH "0")
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckCSourceRuns)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11 -O3 -gencode arch=compute_35,code=sm_35
 -gencode arch=compute_61,code=sm_61 -Xcompiler -funroll-loops")

include_directories(include)
file(GLOB SOURCES "src/*.cu")

CUDA_ADD_EXECUTABLE(stream-cuda ${SOURCES})