cmake_minimum_required (VERSION 3.9)

project(nvbox)

set(CMAKE_CXX_STANDARD 11)

set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")

include_directories("${INCLUDE_DIR}")
include_directories("${SOURCE_DIR}")

file(GLOB ALL_SRC
    "${INCLUDE_DIR}/*.hpp"
    "${SOURCE_DIR}/*.cpp"
    )

add_executable(nvbox "${ALL_SRC}")