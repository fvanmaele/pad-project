#ifndef UPCXX_COMMON_HPP
#define UPCXX_COMMON_HPP
#include <cstddef>
#include <upcxx/upcxx.hpp>

template <typename T>
using dist_ptr = upcxx::dist_object<upcxx::global_ptr<T>>;
using index_t = std::ptrdiff_t;

template <typename T>
T* downcast_dptr(dist_ptr<T> &dp) {
    assert(dp->is_local());
    return dp->local();
}

template <typename T>
T* downcast_gptr(upcxx::global_ptr<T> &gp) {
    assert(gp.is_local());
    return gp.local();
}

#endif // UPCXX_COMMON_HPP