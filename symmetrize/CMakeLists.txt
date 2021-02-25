add_executable(symmetrize-serial-test "serial_test.cpp")
target_link_libraries(symmetrize-serial-test PRIVATE Catch2::Catch2)

add_executable(symmetrize "serial.cpp")

# UPCXX implementation    
add_executable(symmetrize-upcxx "upcxx.cpp")
target_link_libraries(symmetrize-upcxx 
    PRIVATE 
        UPCXX::upcxx)

add_executable(symmetrize-upcxx-skl "upcxx.cpp")
target_link_libraries(symmetrize-upcxx-skl
    PRIVATE 
        UPCXX::upcxx)
target_compile_options(symmetrize-upcxx-skl
    PRIVATE
        -march=skylake)

add_executable(symmetrize-upcxx-knl "upcxx.cpp")
target_link_libraries(symmetrize-upcxx-knl
    PRIVATE 
        UPCXX::upcxx)
target_compile_options(symmetrize-upcxx-knl
    PRIVATE
        -march=knl)


# UPCXX + OpenMP implementation
add_executable(symmetrize-upcxx-openmp "upcxx_openmp.cpp")
target_link_libraries(symmetrize-upcxx-openmp 
    PRIVATE 
        OpenMP::OpenMP_CXX UPCXX::upcxx)

add_executable(symmetrize-upcxx-openmp-skl "upcxx_openmp.cpp")
target_link_libraries(symmetrize-upcxx-openmp-skl
    PRIVATE 
        OpenMP::OpenMP_CXX UPCXX::upcxx)
target_compile_options(symmetrize-upcxx-openmp-skl
    PRIVATE
        -march=skylake)

add_executable(symmetrize-upcxx-openmp-knl "upcxx_openmp.cpp")
target_link_libraries(symmetrize-upcxx-openmp-knl
    PRIVATE 
        OpenMP::OpenMP_CXX UPCXX::upcxx)
target_compile_options(symmetrize-upcxx-openmp-knl
    PRIVATE
        -march=knl)

add_subdirectory("matrix")