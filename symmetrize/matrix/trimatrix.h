#ifndef TRIMATRIX_H
#define TRIMATRIX_H

#include <cassert>
#include <memory>
#include <type_traits>
#include <limits>
#include <iostream>

#include "offsets.h"


namespace asc::pad_ws20::project
{
template <typename T>
class TriMatrix
{
    static_assert(std::is_arithmetic_v<T>);

public:
    typedef T value_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::ptrdiff_t index_t;

    TriMatrix(index_t n) 
        : _n(n), _t(n*(n-1) / 2)
    {
        assert(n >= 1);
        assert(n < std::numeric_limits<index_t>::max() / (n - 1));
        _diag = std::unique_ptr<T>(new T[_n]);

        if (_t > 0) {
            _lower = std::unique_ptr<T>(new T[_t]);
            _upper = std::unique_ptr<T>(new T[_t]);
        }
    }
 
    TriMatrix(T* diag, index_t n, T* lower, T* upper, index_t t)
        : _n(n), _t(t)
    {
        assert(_n >= 1);
        assert(_t == _n*(_n-1) / 2);
        _diag = std::unique_ptr<T>(new T[_n]);

        T* _d = _diag.get();
        for (index_t i = 0; i < _n; ++i) {
            _d[i] = diag[i];
        }
        
        if (_t > 0) {
            _lower = std::unique_ptr<T>(new T[_t]);
            _upper = std::unique_ptr<T>(new T[_t]);

            T* _l = _lower.get();
            T* _u = _upper.get();
            for (index_t k = 0; k < _t; ++k) {
                _l[k] = lower[k];
                _u[k] = upper[k];
            }
        }
    }

    TriMatrix(T* row_major, index_t length)
    {
        assert(length >= 1);
        _n = static_cast<index_t>(std::sqrt(length));
        
        assert(length == _n*_n);
        _t = _n*(_n-1) / 2;
        _diag = std::unique_ptr<T>(new T[_n]);

        if (_n >= 2) {
            assert(_n < std::numeric_limits<index_t>::max() / (_n - 1));
            _lower = std::unique_ptr<T>(new T[_t]);
            _upper = std::unique_ptr<T>(new T[_t]);
        } else {
            _diag.get()[0] = row_major[0];
            return;
        }
        
        T* _l = _lower.get();
        T* _d = _diag.get();
        T* _u = _upper.get();

        for (index_t k = 0; k < length; ++k) {
            index_t j = k % _n;
            index_t i = (k - j) / _n;

            if (i == j) {
                _d[i] = row_major[k];
            } else if (j < i) {
                _l[detail::offset_lower_col_major(i, j, _n)] = row_major[k];
            } else {
                _u[detail::offset_upper_row_major(i, j, _n)] = row_major[k];
            }
        }
    }

    TriMatrix(const TriMatrix&) = delete;
    TriMatrix& operator=(const TriMatrix&) = delete;
    TriMatrix(TriMatrix&&) = default;
    TriMatrix& operator=(TriMatrix&&) = default;
    ~TriMatrix() = default;

    T operator()(index_t i, index_t j) const
    {
        assert(i >= 0 && i < _n);
        assert(j >= 0 && j < _n);

        if (i == j) {
            return _diag.get()[i];
        } else if (i > j) { // row-major order
            return _lower.get()[detail::offset_lower_col_major(i, j, _n)];
        } else { // j > i
            return _upper.get()[detail::offset_upper_row_major(i, j, _n)];
        }
    }

    T& operator()(index_t i, index_t j)
    {
        assert(i >= 0 && i < _n);
        assert(j >= 0 && j < _n);

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

        for (index_t i = 0; i < _t; ++i) {
            T s = (_l[i] + _u[i]) / 2.;
            _l[i] = s;
            _u[i] = s;
        }
    }
    
    T* diag()  { return _diag.get(); }
    T* lower() { return _lower.get(); }
    T* upper() { return _upper.get(); }

    index_t n() const noexcept { return _n; }
    index_t t() const noexcept { return _t; }
    index_t s() const noexcept { return _n + 2*_t; }

private:
    index_t _n;
    index_t _t;

    // For symmetrization of a square matrix, we consider three arrays:
    // - one holding the lower triangle, in col-major order;
    // - one holding the upper triangle, in row-major order;
    // - one holding the diagonal.
    std::unique_ptr<T> _diag;
    std::unique_ptr<T> _lower;
    std::unique_ptr<T> _upper;
};

} // namespace asc::pad_ws20::upcxx

#endif // TRIMATRIX_H
