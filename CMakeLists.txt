# https://cmake.org/cmake/help/latest/policy/CMP0092.html
cmake_minimum_required(VERSION 3.15)
project(pad-project LANGUAGES CXX)

find_package(Catch2 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP 4.0 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(UPCXX_VERBOSE ON)
if (NOT DEFINED ENV{UPCXX_INSTALL})
    set(ENV{UPCXX_INSTALL} "/usr/local/upcxx")  # default install path
endif()
list(APPEND CMAKE_MODULE_PATH "$ENV{UPCXX_INSTALL}/share/cmake")
find_package(UPCXX REQUIRED)

# Lyra Cmdline
find_path(BFGROUP_LYRA_INCLUDE_DIRS "lyra/arg.hpp")
include_directories(${BFGROUP_LYRA_INCLUDE_DIRS})

# Ensure asserts work in Release mode
add_compile_options(-UNDEBUG)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
    )

elseif(CMAKE_CXX_COMPILER MATCHES "icpc.*$") # Intel compiler
    add_compile_options(
        -w3 -diag-disable:remark
        -fp-model precise
        -qopt-zmm-usage=high
        -qopt-report=5
        -qopt-report-phase=vec
        -save-temps -g
    )
endif()

add_subdirectory("reduction")
add_subdirectory("symmetrize")
add_subdirectory("stencil")
