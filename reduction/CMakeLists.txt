# Lyra Cmdline
find_path(BFGROUP_LYRA_INCLUDE_DIRS "lyra/arg.hpp")

add_executable(reduction-serial "serial.cpp")
target_include_directories(reduction-serial 
    PRIVATE 
        ${BFGROUP_LYRA_INCLUDE_DIRS})


# OpenMP implementation (shared systems)
add_executable(reduction-openmp "openmp.cpp")
target_include_directories(reduction-openmp PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(reduction-openmp
    PRIVATE 
        OpenMP::OpenMP_CXX)

add_executable(reduction-openmp-skl "openmp.cpp")
target_include_directories(reduction-openmp-skl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(reduction-openmp-skl 
    PRIVATE 
        OpenMP::OpenMP_CXX)
target_compile_options(reduction-openmp-skl
    PRIVATE 
        -march=skylake)

add_executable(reduction-openmp-knl "openmp.cpp")
target_include_directories(reduction-openmp-knl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(reduction-openmp-knl 
    PRIVATE 
        OpenMP::OpenMP_CXX)
target_compile_options(reduction-openmp-knl
    PRIVATE 
        -march=knl)


# UPCXX implementation (shared and distributed systems)
add_executable(reduction-upcxx "upcxx.cpp")
target_include_directories(reduction-upcxx PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(reduction-upcxx
    PRIVATE 
        UPCXX::upcxx)
  
add_executable(reduction-upcxx-skl "upcxx.cpp")
target_include_directories(reduction-upcxx-skl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(reduction-upcxx-skl
    PRIVATE 
        UPCXX::upcxx)
target_compile_options(reduction-upcxx-skl
    PRIVATE 
        -march=skylake)

add_executable(reduction-upcxx-knl "upcxx.cpp")
target_include_directories(reduction-upcxx-knl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(reduction-upcxx-knl
    PRIVATE 
        UPCXX::upcxx)
target_compile_options(reduction-upcxx-knl
    PRIVATE 
        -march=knl)


# Hybrid implementation (UPCXX for communication, OpenMP for threading)        
add_executable(reduction-hybrid "upcxx_openmp.cpp")
target_include_directories(reduction-hybrid PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(reduction-hybrid
    PRIVATE
    OpenMP::OpenMP_CXX UPCXX::upcxx)

add_executable(reduction-hybrid-skl "upcxx_openmp.cpp")
target_include_directories(reduction-hybrid-skl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(reduction-hybrid-skl
    PRIVATE
    OpenMP::OpenMP_CXX UPCXX::upcxx)
target_compile_options(reduction-hybrid-skl 
    PRIVATE 
        -march=skylake)

add_executable(reduction-hybrid-knl "upcxx_openmp.cpp")
target_include_directories(reduction-hybrid-knl PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(reduction-hybrid-knl
    PRIVATE
    OpenMP::OpenMP_CXX UPCXX::upcxx)
target_compile_options(reduction-hybrid-knl 
    PRIVATE 
        -march=knl)