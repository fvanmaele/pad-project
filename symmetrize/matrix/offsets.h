#ifndef OFFSETS_H
#define OFFSETS_H

#include <cstddef>

namespace asc::pad_ws20::project
{
namespace detail
{
inline std::ptrdiff_t offset_lower_col_major(std::ptrdiff_t i, std::ptrdiff_t j, std::ptrdiff_t n) {
    // 1st summand: expansion of n*(n-1)/2 - (n-1-j)*(n-j)/2
    // 2nd summand: offset for i
    return j*(2*n - 1 - j)/2 + (i - j - 1);
}

inline std::ptrdiff_t offset_upper_row_major(std::ptrdiff_t i, std::ptrdiff_t j, std::ptrdiff_t n) {
    // 1st summand: expansion of n*(n-1)/2 - (n-1-i)*(n-i)/2
    // 2nd summand: offset for j
    return i*(2*n - 1 - i)/2 + (j - i - 1);
}

} // namespace detail
} // namespace asc::pad_ws20::project

#endif // OFFSETS_H