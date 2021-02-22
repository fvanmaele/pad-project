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

OPTION(VECTORIZE_OPTS "Additional options for SIMD (vectorization)" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
    )
    if(VECTORIZE_OPTS)
        add_compile_options(
            -ftree-vectorize
            -fopt-infovec-missed
            -vec-report=5
            -Rpass-missed=loop-vectorize
            -Rpass-analysis=loop-vectorize
            )
    endif()

elseif(CMAKE_CXX_COMPILER MATCHES "icpc.*$") # Intel compiler
    add_compile_options(
        -w3 -diag-disable:remark
        -fp-model precise
        -x${GCC_ARCH}
    )
    if(VECTORIZE_OPTS)
        add_compile_options(
            #-xskylake -axskylake-avx512,mic-avx512 #multi-target architecture
            -qopt-zmm-usage=high
            -qopt-report=5
            -qopt-report-phase=vec
            -save-temps -g
            )
    endif()
endif()

add_subdirectory("reduction")
add_subdirectory("symmetrize")
add_subdirectory("stencil")
