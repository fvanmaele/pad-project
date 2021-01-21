#ifndef TRIMATRIX_H
#define TRIMATRIX_H

#include <cstddef>
#include <memory>
#include <type_traits>
#include <gsl/gsl-lite.hpp>
#include <limits>

namespace asc::pad_ws20::upcxx
{
namespace detail
{
inline ptrdiff_t offset_lower_col_major(ptrdiff_t i, ptrdiff_t j, ptrdiff_t n) {
    // 1st summand: expansion of n*(n-1)/2 - (n-1-j)*(n-j)/2
    // 2nd summand: offset for i
    return j*(2*n - 1 - j)/2 + (i - j - 1);
}

inline ptrdiff_t offset_upper_row_major(ptrdiff_t i, ptrdiff_t j, ptrdiff_t n) {
    // 1st summand: expansion of n*(n-1)/2 - (n-1-i)*(n-i)/2
    // 2nd summand: offset for j
    return i*(2*n - 1 - i)/2 + (j - i - 1);
}
} // namespace detail


template <typename T>
struct TriMatrix
{
    static_assert(std::is_arithmetic_v<T>);
    typedef T value_type;
    typedef ptrdiff_t difference_type;

    TriMatrix(ptrdiff_t n) 
        : _n(n), _t(n*(n-1) / 2)
    {
        gsl_Expects(n >= 1);
        _diag  = std::unique_ptr<T>(new T[_n]);

        if (n >= 2) {
            gsl_Expects(_n < std::numeric_limits<ptrdiff_t>::max() / (_n - 1));
            _lower = std::unique_ptr<T>(new T[_t]);
            _upper = std::unique_ptr<T>(new T[_t]);
        }
    }
 
    T operator()(ptrdiff_t i, ptrdiff_t j) const
    {
        gsl_Expects(i >= 0 && i < _n);
        gsl_Expects(j >= 0 && j < _n);

        if (i == j) {
            return _diag.get()[i];
        } else if (i > j) { // row-major order
            return _lower.get()[detail::offset_lower_col_major(i, j, _n)];
        } else { // j > i
            return _upper.get()[detail::offset_upper_row_major(i, j, _n)];
        }
    }

    T& operator()(ptrdiff_t i, ptrdiff_t j)
    {
        gsl_Expects(i >= 0 && i < _n);
        gsl_Expects(j >= 0 && j < _n);

        if (i == j) {
            return _diag.get()[i];
        } else if (i > j) { // row-major order
            return _lower.get()[detail::offset_lower_col_major(i, j, _n)];
        } else { // j > i
            return _upper.get()[detail::offset_upper_row_major(i, j, _n)];
        }
    }

    void transpose() {
        _lower.swap(_upper);
    }

    void symmetrize() {
        T* _l = _lower.get();
        T* _u = _upper.get();

        for (ptrdiff_t i = 0; i < _t; ++i) {
            T s = (_l[i] + _u[i]) / 2.;
            _l[i] = s;
            _u[i] = s;
        }
    }

    T* diag()  { return _diag.get(); }
    T* lower() { return _lower.get(); }
    T* upper() { return _upper.get(); }

    ptrdiff_t n() const noexcept { return _n; }
    ptrdiff_t t() const noexcept { return _t; }
    ptrdiff_t s() const noexcept { return _n + 2*_t; }

private:
    ptrdiff_t _n;
    ptrdiff_t _t;
    std::unique_ptr<T> _diag;
    std::unique_ptr<T> _lower;
    std::unique_ptr<T> _upper;
};

} // namespace asc::pad_ws20::upcxx

#endif // TRIMATRIX_H
