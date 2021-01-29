
add_executable(reduction-serial "serial.cpp")

add_executable(reduction-numa "numa.cpp")
target_link_libraries(reduction-numa PRIVATE OpenMP::OpenMP_CXX)

add_executable(reduction-upcxx "upcxx.cpp")
target_link_libraries(reduction-upcxx PRIVATE UPCXX::upcxx)
