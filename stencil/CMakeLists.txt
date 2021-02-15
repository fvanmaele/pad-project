# Lyra Cmdline
find_path(BFGROUP_LYRA_INCLUDE_DIRS "lyra/arg.hpp")

add_executable(stencil-serial "serial.cpp")
target_include_directories(stencil-serial PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(stencil-serial PRIVATE fmt::fmt)

add_executable(stencil-openmp "openmp.cpp")
target_include_directories(stencil-openmp PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(stencil-openmp PRIVATE OpenMP::OpenMP_CXX fmt::fmt)

add_executable(stencil-upcxx "upcxx.cpp")
target_include_directories(stencil-upcxx PRIVATE ${BFGROUP_LYRA_INCLUDE_DIRS})
target_link_libraries(stencil-upcxx PRIVATE UPCXX::upcxx fmt::fmt)
