# https://cmake.org/cmake/help/latest/policy/CMP0092.html
cmake_minimum_required(VERSION 3.15)
project(pad-project LANGUAGES CXX)

find_package(fmt REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP 4.0 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT DEFINED GCC_ARCH)
    set(GCC_ARCH "native")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
option(UPCXX_VERBOSE ON)
if (NOT DEFINED ENV{UPCXX_INSTALL})
    set(ENV{UPCXX_INSTALL} "/usr/local/upcxx")  # default install path
endif()
find_package(UPCXX REQUIRED)

OPTION(VECTORIZE_OPTS "Additional options for SIMD (vectorization)" OFF)

if(MSVC)
    add_compile_options(/W4)
    if(STDLIB_DEBUG_MODE)
        add_compile_definitions("$<$<NOT:$<CONFIG:Debug>>:_ITERATOR_DEBUG_LEVEL=1>")
    endif()
    if(VECTORIZE_OPTS)
        add_compile_options(/arch:AVX512)
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic -march=${GCC_ARCH}
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
