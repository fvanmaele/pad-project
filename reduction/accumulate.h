#ifndef ACCUMULATE_H
#define ACCUMULATE_H

#include <algorithm>
#include <type_traits>
#include <cmath>
#include <gsl/gsl-lite.hpp>

namespace PAD
{
namespace detail
{
// Pseudo-concept which checks that S and T are arithmetic types
// which are convertible. It is not checked if conversion between
// S and T results in data loss.
template <typename S, typename T>
inline constexpr bool is_arithmetic_convertible() {
    return std::is_arithmetic_v<S> &&
           std::is_arithmetic_v<T> &&
           std::is_convertible_v<S, T>;
}
} // namespace detail


// Summation with parametrized types for summands and result.
// To avoid rounding errors on large input arrays, the result
// type should be a wider type than the summand type (e.g. double
// for the result, float for the summand).
template <typename S, typename T = S>
T sum(S array[], ptrdiff_t n, T total = 0) {
    static_assert(detail::is_arithmetic_convertible<S, T>());

    for (ptrdiff_t i = 0; i < n; ++i) {
        total += array[i];
    }
    return total;
}

// Summation which recursively breaks the sequence into two halves,
// summing each half, and adding the two sums. This reduces rounding
// errors for large n, and allows parallelization (divide-and-conquer).
// The base case should be chosen sufficiently large to reduce overhead.
// Algorithm: https://en.wikipedia.org/wiki/Pairwise_summation
template <typename S, typename T = S, size_t N = 1000>
T sum_pairwise(S array[], ptrdiff_t n, T total = 0) {
    static_assert(detail::is_arithmetic_convertible<S, T>());
    static_assert(N >= 1);

    while (n > N) {
        ptrdiff_t m = gsl::narrow<ptrdiff_t>(std::floor(n / 2));
        total = sum_pairwise(array, m, total); // array[0] ... array[m-1]

        array += m; // array[m]
        n -= m;     // array+m+(n-m) = array+n
    }
    return sum(array, n, total);
}

// XXX: document
template <typename S, typename T = S>
T sum_kahan(S array[], ptrdiff_t n, S total = 0) {
    T c = 0.0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        T t = total + array[i];

        if (std::fabs(total) >= std::fabs(array[i])) {
            c += (total - t) + array[i];
        } else {
            c += (array[i] - t) + total;
        }
        total = t;
    }
    return total + c;
}


namespace upcxx
{

} // namespace upcxx

} // namespace PAD

#endif // ACCUMULATE_H
