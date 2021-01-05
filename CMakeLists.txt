# https://cmake.org/cmake/help/latest/policy/CMP0092.html
cmake_minimum_required(VERSION 3.15)

project(pad-project LANGUAGES CXX)
find_package(Catch2 REQUIRED)
find_package(gsl-lite REQUIRED)
find_package(benchmark REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(STDLIB_DEBUG_MODE "Enable the debug mode of the standard library" ON)
# TODO enable address sanitizers
#set(SANITIZERS "" CACHE STRING "List of sanitizers to enable (supported values: ASan, UBSan)")

if(MSVC)
    add_compile_options(/W4)

    if(STDLIB_DEBUG_MODE)
        add_compile_definitions("$<$<NOT:$<CONFIG:Debug>>:_ITERATOR_DEBUG_LEVEL=1>")
    endif()

elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)

    if(STDLIB_DEBUG_MODE)
        add_compile_definitions("_GLIBCXX_DEBUG")
        add_compile_definitions("$<IF:$<CONFIG:Debug>,_LIBCPP_DEBUG=1,_LIBCPP_DEBUG=0>")
    endif()

elseif(CMAKE_CXX_COMPILER MATCHES "icpc.*$") # Intel compiler
    add_compile_options(
        -xskylake -axskylake-avx512,mic-avx512 #multi-target architecture
        -qopt-zmm-usage=high
        -qopt-report=5
        -qopt-report-phase=vec
        -save-temps -g
        )

    if (STDLIB_DEBUG_MODE)
        # TODO
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