
add_library(trimatrix INTERFACE)
target_link_libraries(trimatrix INTERFACE gsl::gsl-lite-v1)
target_sources(trimatrix INTERFACE "trimatrix.h")

add_library(matrix INTERFACE)
target_link_libraries(matrix INTERFACE gsl::gsl-lite-v1)
target_sources(matrix INTERFACE "matrix.h")

add_executable(matrix-test "test.cpp")
target_link_libraries(matrix-test PUBLIC matrix trimatrix Catch2::Catch2)
add_executable(matrix-benchmark "benchmark.cpp")
target_link_libraries(matrix-benchmark PUBLIC matrix trimatrix benchmark::benchmark)

enable_testing()
add_test(NAME matrix COMMAND matrix-test)
