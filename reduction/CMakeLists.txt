
add_library(accumulate INTERFACE)
target_link_libraries(accumulate INTERFACE gsl::gsl-lite-v1)
target_sources(accumulate INTERFACE "accumulate.h")

add_executable(accumulate-test "test.cpp")
target_link_libraries(accumulate-test PUBLIC accumulate Catch2::Catch2)

add_executable(accumulate-upcxx-dist "upcxx_dist.cpp")
target_link_libraries(accumulate-upcxx-dist PUBLIC accumulate UPCXX::upcxx)

add_executable(accumulate-upcxx-dist-shared "upcxx_dist_shared.cpp")
target_link_libraries(accumulate-upcxx-dist-shared PUBLIC accumulate UPCXX::upcxx)

add_executable(accumulate-upcxx-smp "upcxx_smp.cpp")
target_link_libraries(accumulate-upcxx-smp PUBLIC accumulate UPCXX::upcxx)

add_executable(accumulate-upcxx-smp-shared "upcxx_smp_shared.cpp")
target_link_libraries(accumulate-upcxx-smp-shared PUBLIC accumulate UPCXX::upcxx)

enable_testing()
add_test(NAME accumulate COMMAND accumulate-test)
