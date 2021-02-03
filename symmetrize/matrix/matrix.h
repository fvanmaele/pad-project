#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <memory>
#include <limits>
#include <type_traits>
#include <gsl/gsl-lite.hpp>
#include <iterator>
#include <iostream>

namespace asc::pad_ws20::upcxx
{
template <typename T>
class SquareMatrix
{
public:
    static_assert(std::is_arithmetic_v<T>);
    typedef T value_type;
    typedef ptrdiff_t difference_type;

    SquareMatrix(ptrdiff_t n)
        : _n(n), _s(n*n)
    {
        gsl_Expects(_n >= 1);
        gsl_Expects(_n < std::numeric_limits<ptrdiff_t>::max() / _n); // overflow check
        _elements = std::unique_ptr<T>(new T[_s]);
    }

    SquareMatrix(T* row_major, ptrdiff_t length) {
        gsl_Expects(length >= 1);
        _n = gsl::narrow_cast<ptrdiff_t>(std::sqrt(length));

        gsl_Expects(length == _n*_n);
        _s = length;
        _elements = std::unique_ptr<T>(new T[_s]);

        T* _e = _elements.get();
        for (ptrdiff_t k = 0; k < length; ++k) {
            _e[k] = row_major[k];
        }
    }

    T operator()(ptrdiff_t i, ptrdiff_t j) const {
        gsl_Expects(i >= 0 && i < _n);
        gsl_Expects(j >= 0 && j < _n);

        return _elements.get()[_n*i + j];
    }

    T& operator()(ptrdiff_t i, ptrdiff_t j) {
        gsl_Expects(i >= 0 && i < _n);
        gsl_Expects(j >= 0 && j < _n);

        return _elements.get()[_n*i + j];
    }

    void transpose() {
        T* elems = _elements.get();

        for (ptrdiff_t i = 0; i < _n ; ++i) {
            // Iterate across lower triangle
            for (ptrdiff_t j = 0; j < i; ++j) {
                ptrdiff_t ij = _n*i + j;
                ptrdiff_t ji = _n*j + i;

                using std::swap;
                swap(elems[ij], elems[ji]);
            }
        }
    }

    void symmetrize() {
        T* elems = _elements.get();

        for (ptrdiff_t i = 0; i < _n; ++i) {
            // Iterate across lower triangle
            for (ptrdiff_t j = 0; j < i; ++j) {
                ptrdiff_t ij = _n*i + j;
                ptrdiff_t ji = _n*j + i;

                T tmp = (elems[ij] + elems[ji]) / 2;
                elems[ij] = tmp;
                elems[ji] = tmp;
            }
        }
    }

    friend std::ostream& dump(std::ostream& stream, const SquareMatrix& M) {
        T* elems = M._elements.get();
        ptrdiff_t last = M._n - 1;
        
        stream << "ELEMS (R-m): " << std::endl;
        for (ptrdiff_t i = 0; i < M._n; ++i) {
            for (ptrdiff_t j = 0; j < M._n; ++j) {
                stream << M(i, j);
                i == last && j == last ? stream << std::endl : stream << " ";
            }
        }
        return stream;
    }

    ptrdiff_t n() const noexcept { return _n; }
    ptrdiff_t t() const noexcept { return _n*(_n - 1) / 2; }
    ptrdiff_t s() const noexcept { return _s; }
    T* elements() { return _elements.get(); }

private:
    ptrdiff_t _n;
    ptrdiff_t _s;
    std::unique_ptr<T> _elements; // row-major order
};

} // namespace asc::pad_ws20::upcxx

#endif // MATRIX_H
