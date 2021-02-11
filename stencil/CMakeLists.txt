
add_executable(stencil-serial "serial.cpp")

add_executable(stencil-openmp "openmp.cpp")
target_link_libraries(stencil-openmp PRIVATE OpenMP::OpenMP_CXX)

add_executable(stencil-upcxx "upcxx.cpp")
target_link_libraries(stencil-upcxx PRIVATE UPCXX::upcxx)

add_subdirectory("FDTD3d_template")