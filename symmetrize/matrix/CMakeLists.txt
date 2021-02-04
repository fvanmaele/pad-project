
add_library(trimatrix INTERFACE)
target_sources(trimatrix INTERFACE "trimatrix.h")

add_library(matrix INTERFACE)
target_sources(matrix INTERFACE "matrix.h")

add_executable(matrix-test "test.cpp")
target_link_libraries(matrix-test PUBLIC Catch2::Catch2)

enable_testing()
add_test(NAME matrix COMMAND matrix-test)
