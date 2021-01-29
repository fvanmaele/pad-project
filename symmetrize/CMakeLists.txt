
add_executable(symmetrize-serial "serial.cpp")

add_executable(symmetrize-numa "numa.cpp")
target_link_libraries(symmetrize-numa PRIVATE OpenMP::OpenMP_CXX)

add_executable(symmetrize-upcxx "upcxx.cpp")
target_link_libraries(symmetrize-upcxx PRIVATE UPCXX::upcxx)
