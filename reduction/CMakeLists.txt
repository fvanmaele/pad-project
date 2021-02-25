
add_executable(reduction "serial.cpp")

# UPCXX implementation
add_executable(reduction-upcxx "upcxx.cpp")
target_link_libraries(reduction-upcxx
    PRIVATE 
        UPCXX::upcxx)
  
add_executable(reduction-upcxx-skl "upcxx.cpp")
target_link_libraries(reduction-upcxx-skl
    PRIVATE 
        UPCXX::upcxx)
target_compile_options(reduction-upcxx-skl
    PRIVATE 
        -march=skylake)

add_executable(reduction-upcxx-knl "upcxx.cpp")
target_link_libraries(reduction-upcxx-knl
    PRIVATE 
        UPCXX::upcxx)
target_compile_options(reduction-upcxx-knl
    PRIVATE 
        -march=knl)


# UPCXX + OpenMP implementation
add_executable(reduction-upcxx-openmp "upcxx_openmp.cpp")
target_link_libraries(reduction-upcxx-openmp
    PRIVATE
        OpenMP::OpenMP_CXX UPCXX::upcxx)

add_executable(reduction-upcxx-openmp-skl "upcxx_openmp.cpp")
target_link_libraries(reduction-upcxx-openmp-skl
    PRIVATE
        OpenMP::OpenMP_CXX UPCXX::upcxx)
target_compile_options(reduction-upcxx-openmp-skl 
    PRIVATE 
        -march=skylake)

add_executable(reduction-upcxx-openmp-knl "upcxx_openmp.cpp")
target_link_libraries(reduction-upcxx-openmp-knl
    PRIVATE
        OpenMP::OpenMP_CXX UPCXX::upcxx)
target_compile_options(reduction-upcxx-openmp-knl 
    PRIVATE 
        -march=knl)