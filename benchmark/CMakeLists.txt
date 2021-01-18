
add_library(benchmark INTERFACE)
target_link_libraries(benchmark INTERFACE fmt::fmt)
target_sources(benchmark INTERFACE "benchmark.h")

add_executable(benchmark-test "test.cpp")
target_link_libraries(benchmark-test benchmark Catch2::Catch2)

