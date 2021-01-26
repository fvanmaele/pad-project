
add_executable(reduction-serial "serial.cpp")

add_executable(reduction-upcxx "upcxx.cpp")
target_link_libraries(reduction-upcxx PRIVATE UPCXX::upcxx)
