cmake_minimum_required (VERSION 3.16)

## Setup dependencies

set(EXTERNAL_LIB_HEADERS
    "depends/argparse/include"
)

include_directories(${EXTERNAL_LIB_HEADERS})

## Setup project

project(
    nvbox
    VERSION 0.0.1
    LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD 20)
endif()

set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories("${INCLUDE_DIR}")
include_directories("${SOURCE_DIR}")

file(GLOB ALL_SRC
    "${INCLUDE_DIR}/*.hpp"
    "${SOURCE_DIR}/*.cpp"
    "${INCLUDE_DIR}/*.cuh"
    "${SOURCE_DIR}/*.cu"
)

add_executable(nvbox "${ALL_SRC}")

target_link_libraries(nvbox)