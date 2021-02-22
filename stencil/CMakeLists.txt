# Lyra Cmdline
find_path(BFGROUP_LYRA_INCLUDE_DIRS "lyra/arg.hpp")

add_executable(stencil-serial "serial.cpp")
target_include_directories(stencil-serial 
    PRIVATE 
        ${BFGROUP_LYRA_INCLUDE_DIRS})

# add_executable(stencil-openmp "openmp.cpp")
# target_include_directories(stencil-openmp PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
# target_link_libraries(stencil-openmp PRIVATE OpenMP::OpenMP_CXX)

add_executable(stencil-upcxx "upcxx.cpp")
target_include_directories(stencil-upcxx PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(stencil-upcxx
    PRIVATE 
        UPCXX::upcxx)

add_executable(stencil-upcxx-skl "upcxx.cpp")
target_include_directories(stencil-upcxx-skl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(stencil-upcxx-skl
    PRIVATE 
        UPCXX::upcxx)
target_compile_options(stencil-upcxx-skl
    PRIVATE 
        -march=skylake)

add_executable(stencil-upcxx-knl "upcxx.cpp")
target_include_directories(stencil-upcxx-knl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(stencil-upcxx-knl
    PRIVATE 
        UPCXX::upcxx)
target_compile_options(stencil-upcxx-knl
    PRIVATE 
        -march=knl)