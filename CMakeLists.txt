cmake_minimum_required (VERSION 3.23)

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

find_package(CUDAToolkit REQUIRED)

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

target_link_libraries(nvbox CUDA::nvml)

## Setup unit tests

option(ENABLE_TESTING "Is testing enabled" ON)

if(ENABLE_TESTING)

  ### Fetch GTest
  include(FetchContent)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
  )

  #### For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  ### Enable testing
  enable_testing()

  file(GLOB ALL_TESTS_SRC "tests/*.cpp")

  add_executable(
    nvbox_test
    "${ALL_TESTS_SRC}"
    )

  target_link_libraries(
    nvbox_test
    gtest_main
    )

  include(GoogleTest)
  gtest_discover_tests(nvbox_test)

endif()