
add_library(accumulate INTERFACE)
target_link_libraries(accumulate INTERFACE gsl::gsl-lite-v1)
target_sources(accumulate INTERFACE "accumulate.h")

add_executable(accumulate-test "test.cpp")
target_link_libraries(accumulate-test PUBLIC accumulate Catch2::Catch2)

add_executable(accumulate-upcxx "upcxx.cpp")
target_link_libraries(accumulate-upcxx PUBLIC accumulate UPCXX::upcxx)

enable_testing()
add_test(NAME accumulate COMMAND accumulate-test)
