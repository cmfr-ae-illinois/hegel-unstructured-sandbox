cmake_minimum_required(VERSION 3.1...3.29)

project(uhegel VERSION 1.0 LANGUAGES C)

file(GLOB_RECURSE sources ./src/*.c ./src/*.h)

include_directories($ENV{PETSC_DIR}/include)
include_directories($ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)

link_directories($ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)

add_executable(uhegel ${sources})

target_link_libraries(uhegel PUBLIC petsc)