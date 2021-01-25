
add_library(reduction INTERFACE)
target_link_libraries(reduction INTERFACE gsl::gsl-lite-v1)
target_sources(reduction INTERFACE "reduction.h")

add_executable(reduction-serial "serial.cpp")
target_link_libraries(reduction-serial PUBLIC reduction)

add_executable(reduction-openmp "smp_openmp.cpp")
target_link_libraries(reduction-openmp PUBLIC reduction PRIVATE OpenMP::OpenMP_CXX Threads::Threads)

add_executable(reduction-upcxx-smp "smp_upcxx.cpp")
target_link_libraries(reduction-upcxx-smp PRIVATE UPCXX::upcxx)

add_executable(reduction-upcxx-smp-shared "smp_upcxx_shared.cpp")
target_link_libraries(reduction-upcxx-smp-shared PRIVATE UPCXX::upcxx)

add_executable(reduction-upcxx-dist "dist_upcxx.cpp")
target_link_libraries(reduction-upcxx-dist PRIVATE UPCXX::upcxx)

add_executable(reduction-upcxx-dist-shared "dist_upcxx_shared.cpp")
target_link_libraries(reduction-upcxx-dist-shared PRIVATE UPCXX::upcxx)

enable_testing()
add_test(NAME reduction COMMAND reduction-test)
