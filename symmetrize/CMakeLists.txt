
add_executable(symmetrize-serial "serial.cpp")
add_executable(symmetrize-serial-test "serial_test.cpp")
target_link_libraries(symmetrize-serial-test PRIVATE Catch2::Catch2)

add_executable(symmetrize-openmp "openmp.cpp")
target_link_libraries(symmetrize-openmp PRIVATE OpenMP::OpenMP_CXX)

add_executable(symmetrize-upcxx "upcxx.cpp")
target_link_libraries(symmetrize-upcxx PRIVATE UPCXX::upcxx)

add_subdirectory("matrix")