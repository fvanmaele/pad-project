
add_executable(reduction-serial "serial.cpp")

add_executable(reduction-openmp "openmp.cpp")
target_link_libraries(reduction-openmp PRIVATE OpenMP::OpenMP_CXX Threads::Threads)

add_executable(reduction-upcxx "upcxx.cpp")
target_link_libraries(reduction-upcxx PRIVATE UPCXX::upcxx)

add_executable(reduction-upcxx-openmp "upcxx_openmp.cpp")
target_link_libraries(reduction-upcxx-openmp PRIVATE UPCXX::upcxx OpenMP::OpenMP_CXX Threads::Threads)
