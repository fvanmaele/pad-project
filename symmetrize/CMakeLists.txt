
add_executable(symmetrize-serial "serial.cpp")

add_executable(symmetrize-openmp "openmp.cpp")
target_link_libraries(symmetrize-openmp PRIVATE OpenMP::OpenMP_CXX)

add_executable(symmetrize-upcxx "upcxx.cpp")
target_link_libraries(symmetrize-upcxx PRIVATE UPCXX::upcxx)

add_subdirectory("matrix")