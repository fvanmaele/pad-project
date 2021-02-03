find_package(Catch2 REQUIRED)
find_package(gsl-lite REQUIRED)

add_library(trimatrix INTERFACE)
target_link_libraries(trimatrix INTERFACE gsl::gsl-lite-v1)
target_sources(trimatrix INTERFACE "trimatrix.h")

add_library(matrix INTERFACE)
target_link_libraries(matrix INTERFACE gsl::gsl-lite-v1)
target_sources(matrix INTERFACE "matrix.h")

add_executable(matrix-test "test.cpp")
target_link_libraries(matrix-test PUBLIC matrix trimatrix Catch2::Catch2)
target_compile_definitions(matrix-test PRIVATE
    # make `gsl_Expects()` throw an exception on precondition failure (so we can write unit tests for failure)
    gsl_CONFIG_CONTRACT_VIOLATION_THROWS
    # check `gsl_ExpectsAudit()` preconditions
    gsl_CONFIG_CONTRACT_CHECKING_AUDIT
)

enable_testing()
add_test(NAME matrix COMMAND matrix-test)
