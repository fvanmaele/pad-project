add_library(trimatrix "trimatrix.cpp" "trimatrix.h")
target_link_libraries(trimatrix PUBLIC gsl::gsl-lite-v1)
add_executable(trimatrix-test "test.cpp")
target_link_libraries(trimatrix-test PUBLIC trimatrix Catch2::Catch2)
add_executable(trimatrix-benchmark "benchmark.cpp")
target_link_libraries(trimatrix-benchmark PUBLIC trimatrix benchmark::benchmark)

enable_testing()
add_test(NAME trimatrix COMMAND trimatrix-test)