#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <memory>
#include <limits>
#include <type_traits>
#include <gsl/gsl-lite.hpp>
#include <utility> // for swap
#include <iterator>

namespace PAD
{
template <typename T>
class SquareMatrix
{
public:
    static_assert(std::is_arithmetic_v<T>);
    typedef T value_type;
    typedef ptrdiff_t difference_type;

    SquareMatrix(ptrdiff_t n)  // elements in row-major order
        : _n(n), _s(n*n)
    {
        gsl_Expects(_n >= 1);
        gsl_Expects(_n < std::numeric_limits<ptrdiff_t>::max() / _n);

        _elements = std::unique_ptr<T>(new T[_s]);
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

    ptrdiff_t n() const noexcept { return _n; }
    ptrdiff_t t() const noexcept { return _n*(_n - 1) / 2; }
    ptrdiff_t s() const noexcept { return _s; }
    T* elements() { return _elements.get(); }

private:
    ptrdiff_t _n;
    ptrdiff_t _s;
    std::unique_ptr<T> _elements; // row-major order
};

}
#endif // MATRIX_H
