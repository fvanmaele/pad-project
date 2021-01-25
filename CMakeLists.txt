# https://cmake.org/cmake/help/latest/policy/CMP0092.html
cmake_minimum_required(VERSION 3.15)
project(pad-project LANGUAGES CXX)

find_package(Catch2 REQUIRED)
find_package(gsl-lite REQUIRED)
find_package(fmt REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP 4.0 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
option(UPCXX_VERBOSE ON)
if (NOT DEFINED ENV{UPCXX_INSTALL})
    set(ENV{UPCXX_INSTALL} "/usr/local/upcxx")  # default install path
endif()
find_package(UPCXX REQUIRED)

option(STDLIB_DEBUG_MODE "Enable the debug mode of the standard library" OFF)
OPTION(AVX_512_VECTORIZE "Optimizations for AVX-512 (Skylake)" OFF)

if(MSVC)
    add_compile_options(/W4)
    if(STDLIB_DEBUG_MODE)
        add_compile_definitions("$<$<NOT:$<CONFIG:Debug>>:_ITERATOR_DEBUG_LEVEL=1>")
    endif()
    if(AVX_512_VECTORIZE)
        add_compile_options(/arch:AVX512)
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
    )
    if(AVX_512_VECTORIZE)
        add_compile_options(
            -march=skylake-avx512 # TODO: other arch (e.g core-avx512, native)
            -ftree-vectorize
            -fopt-infovec-missed
            -vec-report=5
            -Rpass-missed=loop-vectorize
            -Rpass-analysis=loop-vectorize
            )
    endif()
    if(STDLIB_DEBUG_MODE)
        add_compile_definitions("_GLIBCXX_DEBUG")
        add_compile_definitions("$<IF:$<CONFIG:Debug>,_LIBCPP_DEBUG=1,_LIBCPP_DEBUG=0>")
    endif()

elseif(CMAKE_CXX_COMPILER MATCHES "icpc.*$") # Intel compiler
    add_compile_options(
        -w3 -diag-disable:remark
        -fp-model precise
    )
    if(AVX_512_VECTORIZE)
        add_compile_options(
            -xskylake -axskylake-avx512,mic-avx512 #multi-target architecture
            -qopt-zmm-usage=high
            -qopt-report=5
            -qopt-report-phase=vec
            -save-temps -g
            )
    endif()
    if (STDLIB_DEBUG_MODE)
        add_compile_definitions("_GLIBCXX_DEBUG")
    endif()
endif()

# Make `gsl_Expects()` throw an exception on precondition failure (unit tests)
add_compile_definitions(gsl_CONFIG_CONTRACT_VIOLATION_THROWS)
# Enable audit-level checks in Debug mode
add_compile_definitions("$<$<CONFIG:Debug>:gsl_CONFIG_CONTRACT_CHECKING_AUDIT>")
# Disable runtime checks in Release mode
add_compile_definitions("$<$<NOT:$<CONFIG:Debug>>:gsl_CONFIG_CONTRACT_CHECKING_OFF>")

add_subdirectory("reduction")
add_subdirectory("symmetrize")
add_subdirectory("stencil")
add_subdirectory("benchmark")
