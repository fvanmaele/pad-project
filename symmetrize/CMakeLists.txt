
add_executable(symmetrize-serial "serial.cpp")

add_executable(symmetrize-upcxx "upcxx.cpp")
target_link_libraries(symmetrize-upcxx PRIVATE UPCXX::upcxx)
